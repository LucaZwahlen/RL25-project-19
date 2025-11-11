import torch


def grpo_loss(agent, mb_obs, mb_logprobs, mb_actions, mb_values, mb_returns,
              mb_advantages, group_size, ent_coef, vf_coef):
    """
    Group Relative Policy Optimization loss.
    """

    # Forward pass to get new policy logprobs and value estimates
    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(mb_obs, mb_actions.long())

    # Compute policy ratio
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    # Group-relative advantage normalization
    # Reshape advantages into (num_groups, group_size)
    # This ensures that relative advantages are computed within groups
    num_samples = mb_advantages.shape[0]
    num_groups = max(1, num_samples // group_size)

    adv_reshaped = mb_advantages[: num_groups * group_size].view(num_groups, group_size)
    group_mean = adv_reshaped.mean(dim=1, keepdim=True)
    relative_adv = (adv_reshaped - group_mean).view(-1)

    # Handle remainder (if batch size not divisible by group_size)
    if num_samples % group_size != 0:
        remainder_adv = mb_advantages[num_groups * group_size:]
        remainder_adv = remainder_adv - remainder_adv.mean()
        relative_adv = torch.cat([relative_adv, remainder_adv], dim=0)

    # Policy loss (no clipping, relative to group advantage)
    pg_loss = -(ratio * relative_adv.detach()).mean()

    # Optional mild clipping for stability
    # pg_loss = -(torch.clamp(ratio, 0.8, 1.2) * relative_adv.detach()).mean()

    # Value & entropy losses (standard)
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
    entropy_loss = entropy.mean()
    
    # Final GRPO loss
    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

    return loss, pg_loss, v_loss, entropy_loss


def grpo_gae(agent, next_done, next_obs, rewards, dones, values, gamma, gae_lambda, device, num_steps):
    """
    Standard GAE (Generalized Advantage Estimation) used by GRPO.
    Same as PPO, but kept separate for clarity.
    """
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards, device=device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = ~next_done
                nextvalues = next_value
            else:
                nextnonterminal = ~dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
    return advantages, returns
