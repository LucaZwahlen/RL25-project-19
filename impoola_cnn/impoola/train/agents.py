import torch.nn as nn
from torch.distributions.categorical import Categorical

from impoola_cnn.impoola.train.nn import encoder_factory, layer_init_orthogonal


class DQNAgent(nn.Module):
    def __init__(
        self,
        encoder_type,
        envs,
        width_scale=1,
        out_features=256,
        cnn_filters=(16, 32, 32),
        activation="relu",
        use_layer_init_normed=False,
        p_augment=0.0,
        micro_dropout_p=0.0,
    ):
        super().__init__()

        # Encode input images (input as int8, conversion to float32 is done in the encoder forward pass)
        encoder, out_features = encoder_factory(
            encoder_type=encoder_type,
            envs=envs,
            width_scale=width_scale,
            out_features=out_features,
            cnn_filters=cnn_filters,
            activation=activation,
            use_layer_init_normed=use_layer_init_normed,
            p_augment=p_augment,
            micro_dropout_p=micro_dropout_p,
        )
        self.encoder = encoder
        self.out_features = out_features

        self.value = layer_init_orthogonal(
            nn.Linear(out_features, envs.single_action_space.n), std=0.01
        )

    def forward(self, x):
        return self.value(self.encoder(x))

    def get_action(self, x, deterministic=False):
        q_values = self.forward(x)
        if deterministic:
            return q_values.argmax(dim=1)  # same as dist.mode
        else:
            return Categorical(logits=q_values).sample()


class ActorCriticAgent(nn.Module):
    def __init__(
        self,
        encoder_type,
        envs,
        width_scale=1,
        out_features=256,
        cnn_filters=(16, 32, 32),
        activation="relu",
        use_layer_init_normed=False,
        p_augment=0.0,
        micro_dropout_p=0.0,
    ):
        super().__init__()

        # Encode input images (input as int8, conversion to float32 is done in the encoder forward pass)
        encoder, out_features = encoder_factory(
            encoder_type=encoder_type,
            envs=envs,
            width_scale=width_scale,
            out_features=out_features,
            cnn_filters=cnn_filters,
            activation=activation,
            use_layer_init_normed=use_layer_init_normed,
            p_augment=p_augment,
            micro_dropout_p=micro_dropout_p,
        )
        self.encoder = encoder
        self.out_features = out_features

        # Actor head
        actor = nn.Linear(out_features, envs.single_action_space.n)
        # self.actor = layer_init_normed(actor, norm_dim=1, scale=0.1) if use_layer_init_normed else actor
        self.actor = layer_init_orthogonal(actor, std=0.01)

        # Critic head
        critic = nn.Linear(out_features, 1)
        # self.critic = layer_init_normed(critic, norm_dim=1, scale=0.1) if use_layer_init_normed else critic
        self.critic = layer_init_orthogonal(critic, std=1.0)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.actor(hidden), self.critic(hidden)

    def get_value(self, x):
        return self.forward(x)[1]

    def get_pi(self, x):
        return Categorical(logits=self.forward(x)[0])

    def get_action(self, x, deterministic=False):
        pi = self.get_pi(x)
        return pi.sample() if not deterministic else pi.mode

    def get_action_and_value(self, x, action=None):
        raise NotImplementedError


class PPOAgent(ActorCriticAgent):
    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        pi = Categorical(logits=logits)
        if action is None:
            action = pi.sample()
        return action, pi.log_prob(action), pi.entropy(), value, pi.logits

    def get_pi_and_value(self, x):
        logits, value = self.forward(x)
        return Categorical(logits=logits), value


class Vtrace(ActorCriticAgent):
    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        pi = Categorical(logits=logits)
        if action is None:
            action = pi.sample()
        return action, pi.log_prob(action), pi.entropy(), value, pi.logits

    def get_pi_and_value(self, x):
        logits, value = self.forward(x)
        return Categorical(logits=logits), value


import torch
import torch.nn as nn
from torch.distributions import Categorical


class GRPOAgentflavoured(ActorCriticAgent):
    """
    Extension of PPOAgent with GRPO-safe action/value retrieval.
    """

    def get_action_and_value(self, x, action=None, n_actions=None):
        logits, value = self.forward(x)

        # Handle NaN/Inf logits gracefully
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("⚠️ [GRPOAgent] NaN/Inf logits detected – replacing with zeros.")
            logits = torch.zeros_like(logits)

        pi = Categorical(logits=logits)

        # Sample if no action provided
        if action is None:
            action = pi.sample()

        # Clamp actions safely (avoids vectorized_gather_kernel crash)
        if n_actions is not None:
            action = torch.clamp(action, 0, n_actions - 1)

        # Compute standard outputs
        logprob = pi.log_prob(action)
        entropy = pi.entropy()

        # Group-normalized logprob (for GRPO)
        # Optional, for diagnostics; GRPO loss can do its own centering
        logprob = torch.nan_to_num(logprob, nan=0.0, neginf=0.0, posinf=0.0)

        return action, logprob, entropy, value, logits

    def get_pi_and_value(self, x):
        logits, value = self.forward(x)
        logits = torch.nan_to_num(logits, nan=0.0, neginf=0.0, posinf=0.0)
        return Categorical(logits=logits), value


import torch.nn as nn


class GRPOAgent(nn.Module):
    def __init__(
        self,
        encoder_type,
        envs,
        width_scale=1,
        out_features=256,
        cnn_filters=(16, 32, 32),
        activation="relu",
        use_layer_init_normed=False,
        p_augment=0.0,
        micro_dropout_p=0.0,
    ):
        super().__init__()

        encoder, out_features = encoder_factory(
            encoder_type=encoder_type,
            envs=envs,
            width_scale=width_scale,
            out_features=out_features,
            cnn_filters=cnn_filters,
            activation=activation,
            use_layer_init_normed=use_layer_init_normed,
            p_augment=p_augment,
            micro_dropout_p=micro_dropout_p,
        )
        self.encoder = encoder
        self.action_head = layer_init_orthogonal(
            nn.Linear(out_features, envs.single_action_space.n), std=0.01
        )

    def forward(self, x):
        logits = self.action_head(self.encoder(x))
        return Categorical(logits=logits)

    def act(self, x):
        pi = self.forward(x)
        a = pi.sample()
        logp = pi.log_prob(a)
        entropy = pi.entropy()
        return a, logp, entropy
