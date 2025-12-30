import torch
import torch.nn.functional as F

def grpo_loss(
    agent, mb_obs, mb_logprobs, mb_actions, mb_values, 
    mb_returns, mb_advantages, group_size, ent_coef, vf_coef, beta=0.04
):
    # Get current policy distribution and values
    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(mb_obs, mb_actions.long())

    # Ratio and KL (k3 approximation) for trust-region stability
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        approx_kl = (ratio - 1) - logratio

    # Truncate batch to ensure clean group division
    num_samples = mb_advantages.shape[0]
    if num_samples % group_size != 0:
        num_samples = (num_samples // group_size) * group_size
        mb_advantages, ratio, approx_kl = mb_advantages[:num_samples], ratio[:num_samples], approx_kl[:num_samples]

    # GRPO group-relative advantage normalization
    adv_reshaped = mb_advantages.view(-1, group_size)
    group_mean = adv_reshaped.mean(dim=1, keepdim=True)
    group_std = adv_reshaped.std(dim=1, keepdim=True) + 1e-8
    rel_adv = ((adv_reshaped - group_mean) / group_std).view(-1)

    # Policy and KL losses
    pg_loss = -(ratio * rel_adv.detach()).mean()
    kl_penalty = beta * approx_kl.mean()
    
    # Value and Entropy regularizers
    v_loss = 0.5 * ((newvalue.view(-1) - mb_returns) ** 2).mean()
    entropy_loss = entropy.mean()

    loss = pg_loss + kl_penalty - ent_coef * entropy_loss + v_loss * vf_coef
    return loss, pg_loss, v_loss, entropy_loss


def grpo_gae(agent, next_done, next_obs, rewards, dones, values, gamma, gae_lambda, device, num_steps):
    """ Standard GAE implementation for advantage estimation """
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards, device=device)
        lastgaelam = 0
        
        for t in reversed(range(num_steps)):
            nextnonterminal = ~next_done if t == num_steps - 1 else ~dones[t + 1]
            nextvalues = next_value if t == num_steps - 1 else values[t + 1]
            
            # Delta and advantage accumulation
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
        returns = advantages + values
    return advantages, returns


def grpo_consensus_loss(
    agent, mb_obs, mb_logprobs, mb_actions, mb_values,
    mb_returns, mb_advantages, ent_coef, vf_coef, 
    beta=0.04, consensus_coef=0.5, top_p=0.25
):
    # Forward pass with logits for the voting cross-entropy
    _, newlogprob, entropy, newvalue, logits = agent.get_action_and_value(mb_obs, mb_actions.long())

    ratio = (newlogprob - mb_logprobs).exp()
    with torch.no_grad():
        approx_kl = (ratio - 1) - (newlogprob - mb_logprobs)

    # Batch-wide normalization (used when group-seeds aren't available)
    adv_norm = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    # Identify 'Winners' based on top percentile of performance
    k = int(mb_advantages.shape[0] * top_p)
    threshold = torch.topk(adv_norm, k).values[-1]
    winner_mask = (adv_norm >= threshold).float()

    # Consensus Voting: supervised signal to mimic top performers
    ce_loss = F.cross_entropy(logits, mb_actions.long(), reduction='none')
    voting_loss = (ce_loss * winner_mask).sum() / (winner_mask.sum() + 1e-8)

    # Standard PG + KL Stability
    pg_loss = -(ratio * adv_norm.detach()).mean()
    kl_penalty = beta * approx_kl.mean()
    v_loss = 0.5 * ((newvalue.view(-1) - mb_returns) ** 2).mean()

    loss = pg_loss + kl_penalty - ent_coef * entropy.mean() + v_loss * vf_coef + (consensus_coef * voting_loss)
    
    return loss, pg_loss, v_loss, entropy.mean(), voting_loss