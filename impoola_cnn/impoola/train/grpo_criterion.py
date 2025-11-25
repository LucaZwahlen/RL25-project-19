import torch


def grpo_loss(agent, mb_obs, mb_logprobs, mb_actions, mb_values, mb_returns,
              mb_advantages, group_size, ent_coef, vf_coef, beta=0.04):
    """
    Corrected GRPO Loss with KL Penalty and Stable Normalization.
    """
    
    # 1. Forward pass
    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(mb_obs, mb_actions.long())

    # 2. Compute ratios
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    # 3. Calculate Approximate KL Divergence (http://joschu.net/blog/kl-approx.html)
    # GRPO uses this as a penalty instead of PPO's clipping
    with torch.no_grad():
        # approx_kl = logratio.exp() - 1 - logratio  # k1
        approx_kl = (ratio - 1) - logratio         # k3 (cleanrl default)

    # 4. Group-Relative Advantage Normalization
    # We reshape to [Num_Groups, Group_Size]
    # Note: This assumes mb_advantages is ordered by group!
    num_samples = mb_advantages.shape[0]
    
    # Ensure perfect divisibility to avoid "remainder" groups of size 1 (which cause NaN std)
    if num_samples % group_size != 0:
        # Truncate batch to fit group size
        trunc_len = (num_samples // group_size) * group_size
        mb_advantages = mb_advantages[:trunc_len]
        ratio = ratio[:trunc_len]
        approx_kl = approx_kl[:trunc_len]
        num_samples = trunc_len

    num_groups = num_samples // group_size
    
    adv_reshaped = mb_advantages.view(num_groups, group_size)
    
    # Mean centering
    group_mean = adv_reshaped.mean(dim=1, keepdim=True)
    
    # Standard deviation scaling (Optional but recommended for stability in RL)
    # GRPO in LLMs often just subtracts mean, but in RL, scaling helps convergence.
    group_std = adv_reshaped.std(dim=1, keepdim=True) + 1e-8
    
    # Normalized Advantages
    relative_adv = (adv_reshaped - group_mean) / group_std
    relative_adv = relative_adv.view(-1)

    # 5. Policy Loss
    # GRPO Objective: E[ ratio * A_rel ] - beta * KL
    # We want to minimize Loss, so: - (ratio * A_rel) + beta * KL
    pg_loss = -(ratio * relative_adv.detach()).mean()
    
    # Add KL Penalty (The "Trust Region" mechanism of GRPO)
    kl_penalty = beta * approx_kl.mean()

    # 6. Value & Entropy Losses
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
    entropy_loss = entropy.mean()

    # Final Loss
    loss = pg_loss + kl_penalty - ent_coef * entropy_loss + v_loss * vf_coef

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

import torch
import torch.nn.functional as F

def grpo_consensus_loss(agent, mb_obs, mb_logprobs, mb_actions, mb_values, mb_returns,
                        mb_advantages, ent_coef, vf_coef, 
                        beta=0.04,          # KL Penalty strength
                        consensus_coef=0.5, # Strength of the "Voting" pressure
                        top_p=0.25):        # Top 25% of actions are considered "Winners"
    """
    GRPO-style loss that encourages 'Voting' for the best actions in the batch 
    without requiring identical environment seeds.
    """

    # 1. Forward pass
    _, newlogprob, entropy, newvalue, new_logits = agent.get_action_and_value(mb_obs, mb_actions.long())

    # 2. Ratio & KL (The 'Constraint' part)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        approx_kl = (ratio - 1) - logratio

    # 3. Global Advantage Normalization
    # Since we can't group by seed, we group by the whole batch to find relative performance.
    # This tells us: "Who did better than expected compared to the average step?"
    adv_mean = mb_advantages.mean()
    adv_std = mb_advantages.std() + 1e-8
    normalized_adv = (mb_advantages - adv_mean) / adv_std

    # ---------------------------------------------------------
    # THE "CLEVER" PART: Consensus Voting
    # ---------------------------------------------------------
    
    # A. Identify "Winners" (Top P% of advantages)
    # We only want to clone actions that resulted in high positive advantage.
    # We find the advantage threshold for the top_p percentile.
    k = int(mb_advantages.shape[0] * top_p)
    top_val, _ = torch.topk(normalized_adv, k)
    threshold = top_val[-1] # The cutoff score
    
    # Create a mask: 1 for Winners, 0 for Losers
    winner_mask = (normalized_adv >= threshold)

    # B. Calculate Voting Loss (Supervised Learning on Winners)
    # We want the network to increase probability of 'mb_actions' ONLY where winner_mask is True.
    # We use CrossEntropy but mask out the losers.
    
    # new_logits: [Batch, Actions]
    # mb_actions: [Batch]
    ce_loss_all = F.cross_entropy(new_logits, mb_actions.long(), reduction='none')
    
    # Apply mask: We only care about minimizing error for the winners
    voting_loss = (ce_loss_all * winner_mask.float()).sum() / (winner_mask.sum() + 1e-8)
    
    # ---------------------------------------------------------

    # 4. Standard Policy Gradient (GRPO style)
    # We still run standard PG on everyone to discourage bad actions (via negative advantage)
    pg_loss = -(ratio * normalized_adv.detach()).mean()
    
    # 5. KL Penalty (Keep policy stable)
    kl_penalty = beta * approx_kl.mean()
    
    # 6. Value & Entropy
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
    entropy_loss = entropy.mean()

    # Final Loss
    # PG moves mean towards positive advantage
    # Voting Loss explicitly pulls logits towards the "Best" actions
    loss = pg_loss + kl_penalty - ent_coef * entropy_loss + v_loss * vf_coef + (consensus_coef * voting_loss)

    return loss, pg_loss, v_loss, entropy_loss, voting_loss

    