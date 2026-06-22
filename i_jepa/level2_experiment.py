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

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
SAVE_DIR = './checkpoints_ijepa_level2/'
os.makedirs(SAVE_DIR, exist_ok=True)

print("Environment setup complete. Device:", "cuda" if torch.cuda.is_available() else "cpu")

# @title 2. Configuration
class Config:
    batch_size = 512
    learning_rate = 5e-4
    epochs = 10
    img_size = 32
    patch_size = 4  # Smaller patches for CIFAR-10 -> 64 patches total
    embed_dim = 192
    num_heads = 6
    depth = 6
    predictor_depth = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ema_tau_start = 0.996
    ema_tau_end = 1.0

config = Config()

# @title 3. Dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
class FastCIFAR10(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        if train:
            self.data = x_train
            self.targets = y_train.flatten()
        else:
            self.data = x_test
            self.targets = y_test.flatten()
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.data[idx])
        label = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = FastCIFAR10(train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=2)

# @title 4. Model Architecture (ViT-Small inspired)
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class SimpleViT(nn.Module):
    def __init__(self, embed_dim, depth, num_heads):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True, dim_feedforward=embed_dim*4)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.blocks(x))

class Level2_I_JEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, 3, cfg.embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, cfg.embed_dim) * 0.02)
        
        self.context_encoder = SimpleViT(cfg.embed_dim, cfg.depth, cfg.num_heads)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.predictor = SimpleViT(cfg.embed_dim, cfg.predictor_depth, cfg.num_heads)
        self.pred_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

    def update_target_encoder(self, tau):
        with torch.no_grad():
            for c_param, t_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                t_param.data.mul_(tau).add_((1 - tau) * c_param.data)

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x)
        N = patches.shape[1]
        
        # 1. Masking Strategy: Random block masking
        # For simplicity on 8x8 grids (64 patches), we select a random target block
        # target block: roughly 20-30% of patches, context is the rest.
        indices = torch.randperm(N, device=x.device)
        target_size = int(N * 0.25)
        target_idx = indices[:target_size]
        context_idx = indices[target_size:]
        
        # Gather patches
        target_patches = patches[:, target_idx, :] + self.pos_embed[:, target_idx, :]
        context_patches = patches[:, context_idx, :] + self.pos_embed[:, context_idx, :]
        
        # 2. Get Targets
        with torch.no_grad():
            # Pass full image to target encoder to get contextualized targets
            full_targets = self.target_encoder(patches + self.pos_embed)
            target_repr = torch.gather(full_targets, 1, target_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.cfg.embed_dim))
            
        # 3. Context Encoding
        context_repr = self.context_encoder(context_patches)
        
        # 4. Predictor
        # Append target position embeddings to context
        target_pos = self.pos_embed[:, target_idx, :].expand(B, -1, -1)
        pred_input = torch.cat([context_repr, target_pos], dim=1)
        pred_out = self.predictor(pred_input)
        
        # Extract predictions for the target positions (the last 'target_size' tokens)
        predicted_target_repr = self.pred_proj(pred_out[:, -target_size:, :])
        
        return predicted_target_repr, target_repr

model = Level2_I_JEPA(config).to(config.device)
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.04)

# @title 5. Training Loop
print("Starting Training on", config.device)
model.train()

for epoch in range(config.epochs):
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
    
    # Cosine annealing for EMA tau
    tau = config.ema_tau_start + (config.ema_tau_end - config.ema_tau_start) * (epoch / config.epochs)
    
    for img, _ in pbar:
        img = img.to(config.device)
        optimizer.zero_grad()
        
        pred_target, true_target = model(img)
        
        # Smooth L1 Loss is often preferred for stability
        loss = F.smooth_l1_loss(pred_target, true_target)
        
        loss.backward()
        optimizer.step()
        
        model.update_target_encoder(tau)
        
        epoch_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Tau": f"{tau:.4f}"})
        
    print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ijepa_level2_epoch_{epoch+1}.pt"))
