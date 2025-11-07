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

