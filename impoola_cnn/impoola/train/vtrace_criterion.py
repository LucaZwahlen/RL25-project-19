from contextlib import nullcontext

import torch
from torch.distributions import Categorical


def compute_vtrace_targets(rewards, dones, values, bootstrap_value, behavior_logits, target_logits, gamma, rho_bar,
                           c_bar, actions):
    t, n = rewards.shape
    k = behavior_logits.shape[-1]
    with torch.no_grad():
        bl = behavior_logits.view(t * n, k)
        tl = target_logits.view(t * n, k)
        a = actions.view(t * n).long()
        logp_b = (bl - bl.logsumexp(-1, keepdim=True)).gather(1, a.view(-1, 1)).squeeze(1).view(t, n)
        logp_t = (tl - tl.logsumexp(-1, keepdim=True)).gather(1, a.view(-1, 1)).squeeze(1).view(t, n)
        rho = (logp_t - logp_b).exp()
        rho_bar_t = torch.minimum(rho, torch.as_tensor(rho_bar, device=rho.device, dtype=rho.dtype))
        c_bar_t = torch.minimum(rho, torch.as_tensor(c_bar, device=rho.device, dtype=rho.dtype))
        not_done = (~dones).float()
        v = values
        v_tp1 = torch.cat([v[1:], bootstrap_value.view(1, n)], dim=0)
        deltas = rho_bar_t * (rewards + gamma * v_tp1 * not_done - v)
        vs = torch.zeros_like(v)
        vs_tp1 = bootstrap_value
        for i in range(t - 1, -1, -1):
            corr = gamma * c_bar_t[i] * (vs_tp1 - v_tp1[i]) * not_done[i]
            vs_i = v[i] + deltas[i] + corr
            vs[i] = vs_i
            vs_tp1 = vs_i
        v_tp1_pg = torch.cat([vs[1:], bootstrap_value.view(1, n)], dim=0)
        pg_adv = rho_bar_t * (rewards + gamma * v_tp1_pg * not_done - v)
    return vs, pg_adv


def vtrace_loss(agent, mb_obs, mb_actions, mb_behavior_logits, mb_values, mb_rewards, mb_dones, mb_bootstrap_value,
                gamma, rho_bar, c_bar, ent_coef, vf_coef, amp=False):
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (amp and mb_obs.is_cuda) else nullcontext()
    with amp_ctx:
        pi, values_new = agent.get_pi_and_value(mb_obs)
        target_logits = pi.logits
        values_new = values_new.view_as(mb_values)
        vs, pg_adv = compute_vtrace_targets(
            rewards=mb_rewards,
            dones=mb_dones,
            values=mb_values,
            bootstrap_value=mb_bootstrap_value,
            behavior_logits=mb_behavior_logits,
            target_logits=target_logits.view_as(mb_behavior_logits),
            gamma=gamma,
            rho_bar=rho_bar,
            c_bar=c_bar,
            actions=mb_actions.view(-1)
        )
        dist = Categorical(logits=target_logits.view(-1, target_logits.shape[-1]))
        logp = dist.log_prob(mb_actions.view(-1)).view_as(pg_adv)
        entropy = dist.entropy().view_as(pg_adv)
        policy_loss = -(pg_adv.detach() * logp).mean()
        value_loss = 0.5 * (values_new - vs.detach()).pow(2).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
    return loss, policy_loss, value_loss, entropy_loss


def vtrace_unrolls(agent, next_done, next_obs, rewards, dones, values, behavior_logits, device, unroll_length):
    with torch.no_grad():
        bootstrap_value = agent.get_pi_and_value(next_obs)[1].reshape(-1)
    return bootstrap_value
