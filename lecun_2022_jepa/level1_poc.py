# @title 1. Environment Setup
# !pip install torch torchvision tqdm wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import os

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Optional: Mount Google Drive if running in Colab
# from google.colab import drive
# drive.mount('/content/drive')
# SAVE_DIR = '/content/drive/MyDrive/paper_implementations/lecun_2022_jepa/'
# os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_DIR = './checkpoints/'
os.makedirs(SAVE_DIR, exist_ok=True)

print("Environment setup complete. Device:", "cuda" if torch.cuda.is_available() else "cpu")

# @title 2. Configuration
class Config:
    batch_size = 256
    learning_rate = 1e-3
    epochs = 5
    latent_dim = 64
    hidden_dim = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variance_weight = 1.0  # Weight for variance regularization to prevent collapse
    target_std = 1.0

config = Config()

# @title 3. Dataset
# We need two "views" of the same image. We'll use a clean image and a noisy/rotated version.
class JepaMnistDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.base_dataset = torchvision.datasets.MNIST(
            root='./data', train=train, download=True
        )
        self.clean_transform = transforms.ToTensor()
        # Transform for View 2 (e.g. random rotation or noise)
        self.aug_transform = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        view_1 = self.clean_transform(img)   # "Clean" or first view
        view_2 = self.aug_transform(img)     # "Augmented" or second view
        return view_1, view_2, label

train_dataset = JepaMnistDataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# @title 4. Model Architecture (Toy JEPA)
class Encoder(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        # In a full JEPA, this would also take an action 'z'. 
        # Here we just predict sx -> sy directly (implicit action)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, sx):
        return self.net(sx)

class ToyJEPA(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.predictor = Predictor(latent_dim=latent_dim, hidden_dim=hidden_dim)

model = ToyJEPA(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# @title 5. Training Loop
def variance_loss(x, target_std=1.0, eps=1e-4):
    """
    Computes a hinge loss to ensure the standard deviation of each feature 
    across the batch is at least `target_std`. This prevents representation collapse.
    """
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(F.relu(target_std - std))

# Sanity check: Ensure shapes are correct before training
print("Running Sanity Check...")
_v1, _v2, _ = next(iter(train_loader))
_v1, _v2 = _v1.to(config.device), _v2.to(config.device)
_sx = model.encoder(_v1)
_sy = model.encoder(_v2)
_pred_sy = model.predictor(_sx)
assert _sx.shape == (config.batch_size, config.latent_dim), "Encoder shape mismatch"
assert _pred_sy.shape == (config.batch_size, config.latent_dim), "Predictor shape mismatch"
print("Sanity Check Passed!\n")

print("Starting Training...")
model.train()
for epoch in range(config.epochs):
    epoch_loss = 0.0
    epoch_pred_loss = 0.0
    epoch_var_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
    for view_1, view_2, _ in pbar:
        view_1, view_2 = view_1.to(config.device), view_2.to(config.device)
        
        optimizer.zero_grad()
        
        # 1. Encode both views
        sx = model.encoder(view_1)
        sy = model.encoder(view_2)
        
        # 2. Predict view 2 from view 1
        pred_sy = model.predictor(sx)
        
        # 3. Calculate Loss
        # Prediction Loss: Mean Squared Error in Latent Space
        pred_loss = F.mse_loss(pred_sy, sy.detach()) # Detach sy to prevent collapse through target encoder
        
        # Variance Regularization: Keep embeddings spread out
        var_loss_sx = variance_loss(sx, target_std=config.target_std)
        var_loss_sy = variance_loss(sy, target_std=config.target_std)
        var_loss = var_loss_sx + var_loss_sy
        
        # Total Loss
        loss = pred_loss + config.variance_weight * var_loss
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_pred_loss += pred_loss.item()
        epoch_var_loss += var_loss.item()
        
        pbar.set_postfix({"Pred Loss": f"{pred_loss.item():.4f}", "Var Loss": f"{var_loss.item():.4f}"})
        
    print(f"Epoch {epoch+1} Summary | Total Loss: {epoch_loss/len(train_loader):.4f} | Pred Loss: {epoch_pred_loss/len(train_loader):.4f} | Var Loss: {epoch_var_loss/len(train_loader):.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"toy_jepa_epoch_{epoch+1}.pt"))

# @title 6. Evaluation / Visualization (Sanity Check)
print("Evaluating representations...")
model.eval()
with torch.no_grad():
    v1, v2, _ = next(iter(train_loader))
    sx = model.encoder(v1.to(config.device))
    print("Batch Standard Deviation (should be > 0.1 to avoid collapse):", sx.std(dim=0).mean().item())
