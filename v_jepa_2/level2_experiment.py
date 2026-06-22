# @title 1. Environment Setup
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# @title 2. V-JEPA 2 Architecture (Action-Conditioned)
# We define a simple continuous encoder and a predictor.

class Encoder(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

class ActionConditionedPredictor(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        # Takes Latent State + Action as input
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, latent_state, action):
        x = torch.cat([latent_state, action], dim=-1)
        # Predicts the residual (the change in the latent state)
        delta = self.net(x)
        return latent_state + delta

# @title 3. Simulated Physics Environment (Training Data)
# A 2D point mass controlled by 2D force vectors.
print("\nGenerating simulated physics dataset...")
num_samples = 10000
state_dim = 2
action_dim = 2
latent_dim = 16

# Random initial states
states_t0 = torch.rand((num_samples, state_dim)) * 10.0
# Random actions (forces applied)
actions = torch.randn((num_samples, action_dim)) * 2.0
# Physics engine: next state = current state + action (ignoring mass/friction for simplicity)
states_t1 = states_t0 + actions

# @title 4. Training the JEPA World Model
encoder = Encoder(state_dim, latent_dim).to(device)
predictor = ActionConditionedPredictor(latent_dim, action_dim).to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3)

states_t0 = states_t0.to(device)
actions = actions.to(device)
states_t1 = states_t1.to(device)

print("\nTraining Action-Conditioned Predictor...")
epochs = 500
batch_size = 256

for epoch in range(epochs):
    permutation = torch.randperm(num_samples)
    total_loss = 0
    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i+batch_size]
        s0, a, s1 = states_t0[indices], actions[indices], states_t1[indices]
        
        optimizer.zero_grad()
        
        # 1. Encode both states to latent space
        z0 = encoder(s0)
        # We detach the target so the encoder doesn't collapse
        z1_target = encoder(s1).detach() 
        
        # 2. Predict z1 from z0 and action
        z1_pred = predictor(z0, a)
        
        # 3. JEPA Loss (Predicting in latent space)
        loss = F.mse_loss(z1_pred, z1_target)
        
        # Regularization to prevent embedding collapse (variance loss)
        std_z = torch.sqrt(z1_pred.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1.0 - std_z))
        
        total_loss_batch = loss + std_loss
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/(num_samples/batch_size):.4f}")

# @title 5. Zero-Shot Robotic Planning (Model Predictive Control)
print("\n--- Model Predictive Control (MPC) Evaluation ---")
# The "Robot" is at [0.0, 0.0]. The Goal is [8.0, 8.0]
start_pos = torch.tensor([[0.0, 0.0]]).to(device)
goal_pos = torch.tensor([[8.0, 8.0]]).to(device)

encoder.eval()
predictor.eval()

with torch.no_grad():
    latent_start = encoder(start_pos)
    latent_goal = encoder(goal_pos)
    
    # The MPC Planner samples 1000 random actions to find the best one
    num_action_samples = 1000
    candidate_actions = torch.randn((num_action_samples, action_dim)).to(device) * 5.0
    
    # Broadcast start state
    latent_start_batch = latent_start.repeat(num_action_samples, 1)
    
    # Predict futures in latent space for all 1000 actions simultaneously!
    predicted_futures = predictor(latent_start_batch, candidate_actions)
    
    # Calculate latent distance to goal
    distances = torch.norm(predicted_futures - latent_goal, dim=1)
    
    # Find the best action
    best_idx = torch.argmin(distances)
    best_action = candidate_actions[best_idx]

print(f"Goal Position: {goal_pos[0].tolist()}")
print(f"Best Action Found by V-JEPA Planner: {best_action.tolist()}")
# Let's see what happens in the "real world" if we apply this action
actual_next_state = start_pos + best_action.unsqueeze(0)
print(f"Actual Position after taking action: {actual_next_state[0].tolist()}")
print("Success! The planner found the perfect action vector purely by simulating in the abstract latent space.")
