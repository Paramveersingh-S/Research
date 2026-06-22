import torch
import torch.nn.functional as F

print("--- DeepSeek-R1 GRPO Loss Simulation (Level 1 POC) ---")

def calculate_grpo_loss(
    pi_theta_logprobs,  # [G] Log probs of the generated outputs under CURRENT policy
    pi_ref_logprobs,    # [G] Log probs of the generated outputs under REFERENCE policy (frozen)
    rewards,            # [G] Objective reward scores for each of the G outputs
    epsilon=0.2,        # PPO Clipping parameter
    beta=0.01           # KL Divergence penalty coefficient
):
    """
    Computes the Group Relative Policy Optimization (GRPO) loss for a single prompt.
    G is the group size (number of sampled outputs for the prompt).
    """
    G = rewards.shape[0]
    
    # 1. Compute Group Relative Advantages
    # Instead of a neural Critic model, GRPO simply standardizes the rewards within the group.
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-8
    advantages = (rewards - mean_reward) / std_reward
    
    print(f"Raw Rewards: {rewards.tolist()}")
    print(f"Advantages: {advantages.tolist()}\n")
    
    # 2. Calculate Importance Sampling Ratios
    # ratio = exp(log_prob_current - log_prob_old)
    # Note: In standard PPO, we use log_prob_old. 
    # Here for simplicity of the POC, we assume the old policy is the reference policy.
    # (In real GRPO, old policy is from the start of the PPO step, ref policy is from the start of training)
    ratios = torch.exp(pi_theta_logprobs - pi_ref_logprobs)
    
    # 3. PPO Surrogate Loss (Clipped Objective)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages
    
    # We want to MAXIMIZE advantage, so loss is negative surrogate
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 4. KL Divergence Penalty
    # We penalize the policy if it strays too far from the reference model.
    # DeepSeek uses an unbiased estimator: D_KL = exp(ref - theta) - (ref - theta) - 1
    # For simplicity, a standard approximation is: log_prob_theta - log_prob_ref
    kl_div = pi_theta_logprobs - pi_ref_logprobs
    kl_penalty = beta * kl_div.mean()
    
    # Total Loss
    total_loss = policy_loss + kl_penalty
    
    print(f"Policy Loss: {policy_loss.item():.4f}")
    print(f"KL Penalty:  {kl_penalty.item():.4f}")
    print(f"Total Loss:  {total_loss.item():.4f}\n")
    
    return total_loss

# --- Simulation Setup ---
G = 4 # We sample 4 different answers for the prompt "What is 2+2?"

# Let's say the reference model assigned these log probabilities to the 4 sequences
pi_ref_logprobs = torch.tensor([-2.0, -1.5, -2.5, -1.8])

# Let's say our CURRENT model (after some updates) assigns these log probs
pi_theta_logprobs = torch.tensor([-1.9, -1.0, -2.8, -1.8], requires_grad=True)

# The objective rule-based reward function (e.g. checking if "4" is in the answer)
# Output 2 got it right, the others got it wrong.
rewards = torch.tensor([0.0, 1.0, 0.0, 0.0])

# Compute Loss
loss = calculate_grpo_loss(pi_theta_logprobs, pi_ref_logprobs, rewards)

# Backpropagate to simulate updating the model weights
loss.backward()

print("Gradients on pi_theta_logprobs (how the model should change its probabilities):")
print(pi_theta_logprobs.grad)
print("\nInterpretation:")
print("Negative gradient means the model should INCREASE the probability of that output.")
print("Notice how the gradient for Output 2 is highly negative (encourage!), while others are positive (discourage!).")
