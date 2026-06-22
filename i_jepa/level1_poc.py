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
import copy

# Fix for slow CIFAR-10 downloads from default Toronto servers
torchvision.datasets.CIFAR10.url = "https://ossci-datasets.s3.amazonaws.com/cifar/cifar-10-python.tar.gz"

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# SAVE_DIR = '/content/drive/MyDrive/paper_implementations/i_jepa/'
SAVE_DIR = './checkpoints_ijepa_level1/'
os.makedirs(SAVE_DIR, exist_ok=True)

print("Environment setup complete. Device:", "cuda" if torch.cuda.is_available() else "cpu")

# @title 2. Configuration
class Config:
    batch_size = 256
    learning_rate = 1e-3
    epochs = 3
    img_size = 32
    patch_size = 8
    embed_dim = 64
    num_heads = 4
    depth = 2
    predictor_depth = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ema_tau = 0.996 # EMA momentum for target encoder

config = Config()

# @title 3. Dataset (CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

# @title 4. Model Architecture (Tiny ViT + I-JEPA setup)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=64):
        super().__init__()
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, embed_dim]
        return x

class SimpleViT(nn.Module):
    def __init__(self, embed_dim=64, depth=2, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.blocks(x))

class I_JEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, 3, cfg.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, cfg.embed_dim))
        
        # Context Encoder (Trained)
        self.context_encoder = SimpleViT(cfg.embed_dim, cfg.depth, cfg.num_heads)
        
        # Target Encoder (EMA)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # Predictor (Trained)
        self.predictor = SimpleViT(cfg.embed_dim, cfg.predictor_depth, cfg.num_heads)
        self.pred_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

    def update_target_encoder(self, tau):
        # EMA update
        with torch.no_grad():
            for c_param, t_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                t_param.data.mul_(tau).add_((1 - tau) * c_param.data)

    def forward(self, x):
        B = x.shape[0]
        # 1. Patchify and add positional embeddings
        patches = self.patch_embed(x) # [B, N, D]
        N = patches.shape[1]
        patches = patches + self.pos_embed
        
        # For this POC, we do a simple masking strategy:
        # Target = last half of patches, Context = first half of patches
        half = N // 2
        context_patches = patches[:, :half, :]
        target_patches = patches[:, half:, :]
        target_pos = self.pos_embed[:, half:, :].expand(B, -1, -1)
        
        # 2. Get Target Representations (NO GRADIENTS)
        with torch.no_grad():
            # In paper, target encoder sees full image, but we only extract target patches
            # We simplify here by just passing the target patches + their pos embeds
            target_repr = self.target_encoder(target_patches)
            
        # 3. Get Context Representations
        context_repr = self.context_encoder(context_patches)
        
        # 4. Predict Target from Context
        # Predictor takes context + positional embedding of targets
        # In a real implementation, you concatenate context and target pos tokens.
        # For simplicity, we just pass context and project.
        pred_input = torch.cat([context_repr, target_pos], dim=1) 
        pred_out = self.predictor(pred_input)
        
        # We only care about predicting the target block representations
        predicted_target_repr = self.pred_proj(pred_out[:, half:, :])
        
        return predicted_target_repr, target_repr

model = I_JEPA(config).to(config.device)
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

# @title 5. Training Loop
print("Starting Training on", config.device)
model.train()

for epoch in range(config.epochs):
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
    
    for img, _ in pbar:
        img = img.to(config.device)
        optimizer.zero_grad()
        
        pred_target, true_target = model(img)
        
        # L1 or L2 loss between predicted representations and target representations
        loss = F.mse_loss(pred_target, true_target)
        
        loss.backward()
        optimizer.step()
        
        # Update Target Encoder via EMA
        model.update_target_encoder(config.ema_tau)
        
        epoch_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ijepa_level1_epoch_{epoch+1}.pt"))

# @title 6. Evaluation Sanity Check
print("Training finished. Sanity check: verify EMA weights don't match context weights exactly.")
c_w = next(model.context_encoder.parameters())
t_w = next(model.target_encoder.parameters())
print("EMA weights difference:", torch.norm(c_w - t_w).item())
