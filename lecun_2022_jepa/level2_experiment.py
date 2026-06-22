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
SAVE_DIR = './checkpoints_level2/'
os.makedirs(SAVE_DIR, exist_ok=True)

print("Environment setup complete. Device:", "cuda" if torch.cuda.is_available() else "cpu")

# @title 2. Configuration
class Config:
    batch_size = 256
    learning_rate = 3e-4
    epochs = 10
    latent_dim = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variance_weight = 1.0
    target_std = 1.0

config = Config()

# @title 3. Dataset (CIFAR-10)
# Here we mimic a simple self-supervised task:
# View 1: Left half of the image
# View 2: Right half of the image (or heavily augmented version)
class JepaCifarDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.base_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True
        )
        self.to_tensor = transforms.ToTensor()
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.clean = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        # JEPA standard self-supervised setup: two augmented views
        view_1 = self.aug(img)
        view_2 = self.aug(img)
        return view_1, view_2, label

train_dataset = JepaCifarDataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)

# @title 4. Model Architecture (Tiny ResNet Encoder)
class SimpleConvEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # Extremely simplified CNN for fast execution
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, sx):
        return self.net(sx)

class Level2JEPA(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = SimpleConvEncoder(latent_dim=latent_dim)
        # Using a momentum encoder (EMA) for the target is standard in JEPA/BYOL
        # Here we skip EMA for simplicity and just use the same weights, but detach gradients.
        self.predictor = Predictor(latent_dim=latent_dim)

model = Level2JEPA(latent_dim=config.latent_dim).to(config.device)
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

# @title 5. Training Loop
def variance_loss(x, target_std=1.0, eps=1e-4):
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(F.relu(target_std - std))

print("Starting Training on", config.device)
model.train()

for epoch in range(config.epochs):
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
    for view_1, view_2, _ in pbar:
        view_1, view_2 = view_1.to(config.device), view_2.to(config.device)
        
        optimizer.zero_grad()
        
        sx = model.encoder(view_1)
        sy = model.encoder(view_2)
        
        pred_sy = model.predictor(sx)
        
        # Asymmetrical prediction to avoid trivial collapse (along with variance loss)
        pred_loss = F.mse_loss(pred_sy, sy.detach())
        
        var_loss = variance_loss(sx, target_std=config.target_std) + variance_loss(sy, target_std=config.target_std)
        
        loss = pred_loss + config.variance_weight * var_loss
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Pred": f"{pred_loss.item():.4f}", "Var": f"{var_loss.item():.4f}"})
        
    print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"level2_jepa_epoch_{epoch+1}.pt"))

# @title 6. Evaluation
print("Evaluating representations...")
model.eval()
with torch.no_grad():
    v1, v2, _ = next(iter(train_loader))
    sx = model.encoder(v1.to(config.device))
    print("Batch Standard Deviation (should be ~1.0):", sx.std(dim=0).mean().item())
