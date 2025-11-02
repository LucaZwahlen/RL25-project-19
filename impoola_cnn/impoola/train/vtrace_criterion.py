import torch


def compute_vtrace_targets(rewards,
                           dones,
                           values,
                           bootstrap_value,
                           behavior_logits,
                           target_logits,
                           gamma,
                           rho_bar,
                           c_bar,
                           actions):
    T, N = rewards.shape
    K = behavior_logits.shape[-1]

    with torch.no_grad():
        behavior = torch.distributions.Categorical(logits=behavior_logits.view(T * N, K))
        target = torch.distributions.Categorical(logits=target_logits.view(T * N, K))
        a = actions.view(T * N).long()

        logp_b = behavior.log_prob(a).view(T, N)
        logp_t = target.log_prob(a).view(T, N)

        rho = (logp_t - logp_b).exp()
        rho_bar_t = torch.clamp(rho, max=float(rho_bar))
        c_bar_t = torch.clamp(rho, max=float(c_bar))

        not_done = (~dones).float()
        V = values
        V_tp1 = torch.cat([V[1:], bootstrap_value.view(1, N)], dim=0)

        deltas = rho_bar_t * (rewards + gamma * V_tp1 * not_done - V)

        vs = torch.zeros_like(V)
        vs_tp1 = bootstrap_value  # [N]
        for t in reversed(range(T)):
            correction = gamma * c_bar_t[t] * (vs_tp1 - V_tp1[t]) * not_done[t]
            vs[t] = V[t] + deltas[t] + correction
            vs_tp1 = vs[t]

        v_tp1_for_pg = torch.cat([vs[1:], bootstrap_value.view(1, N)], dim=0)
        pg_adv = rho_bar_t * (rewards + gamma * v_tp1_for_pg * not_done - V)

    return vs, pg_adv


def vtrace_loss(agent,
                mb_obs,
                mb_actions,
                mb_behavior_logits,
                mb_values,
                mb_rewards,
                mb_dones,
                mb_bootstrap_value,
                gamma,
                rho_bar,
                c_bar,
                ent_coef,
                vf_coef):
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

    dist = torch.distributions.Categorical(logits=target_logits.view(-1, target_logits.shape[-1]))
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
