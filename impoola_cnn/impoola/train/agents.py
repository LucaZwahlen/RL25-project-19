import torch.nn as nn
from impoola.train.nn import encoder_factory, layer_init_orthogonal
from torch.distributions.categorical import Categorical


class DQNAgent(nn.Module):
    def __init__(
            self,
            encoder_type,
            envs,
            width_scale=1, out_features=256, cnn_filters=(16, 32, 32), activation='relu',
            use_layer_init_normed=False,

    ):
        super().__init__()

        # Encode input images (input as int8, conversion to float32 is done in the encoder forward pass)
        encoder, out_features = encoder_factory(
            encoder_type=encoder_type,
            envs=envs,
            width_scale=width_scale, out_features=out_features, cnn_filters=cnn_filters, activation=activation,
            use_layer_init_normed=use_layer_init_normed,

        )
        self.encoder = encoder
        self.out_features = out_features

        self.value = layer_init_orthogonal(nn.Linear(out_features, envs.single_action_space.n), std=0.01)

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
            width_scale=1, out_features=256, cnn_filters=(16, 32, 32), activation='relu',
            use_layer_init_normed=False,


    ):
        super().__init__()

        # Encode input images (input as int8, conversion to float32 is done in the encoder forward pass)
        encoder, out_features = encoder_factory(
            encoder_type=encoder_type,
            envs=envs,
            width_scale=width_scale, out_features=out_features, cnn_filters=cnn_filters, activation=activation,
            use_layer_init_normed=use_layer_init_normed,
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
