# @title 1. Environment Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
from tqdm import tqdm

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# @title 2. Configuration
class Config:
    batch_size = 32
    num_frames = 8
    img_size = 28
    patch_size = 14
    time_patch_size = 2 # How many frames grouped into one patch
    latent_dim = 64
    hidden_dim = 128
    num_heads = 4
    num_layers = 2
    epochs = 5
    lr = 1e-3
    ema_momentum = 0.99
    variance_weight = 1.0
    target_std = 1.0

config = Config()

# Calculate sequence length based on 3D patches
assert config.num_frames % config.time_patch_size == 0
assert config.img_size % config.patch_size == 0
T_patches = config.num_frames // config.time_patch_size
H_patches = config.img_size // config.patch_size
W_patches = config.img_size // config.patch_size
num_patches = T_patches * H_patches * W_patches

# @title 3. Moving MNIST Simulation
class MovingMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=8):
        self.base_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        self.num_frames = num_frames
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        img = transforms.ToTensor()(img) # [1, 28, 28]
        
        # Simulate motion by translating the image slightly across frames
        frames = []
        for t in range(self.num_frames):
            # Translate right and down by 't' pixels
            transform = transforms.AffineTransform(translation=(t, t))
            frame = transform(img)
            frames.append(frame)
            
        video = torch.stack(frames, dim=1) # [1, T, 28, 28]
        return video

train_dataset = MovingMNISTDataset(num_frames=config.num_frames)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

# @title 4. V-JEPA Architecture
class TubeletEmbedding(nn.Module):
    """3D Patch Embedding (Tubelets)"""
    def __init__(self, in_channels=1, embed_dim=64, patch_size=14, time_patch_size=2):
        super().__init__()
        # Conv3d takes [B, C, T, H, W]
        self.proj = nn.Conv3d(in_channels, embed_dim, 
                              kernel_size=(time_patch_size, patch_size, patch_size), 
                              stride=(time_patch_size, patch_size, patch_size))
        
    def forward(self, x):
        x = self.proj(x) # [B, embed_dim, T_patches, H_patches, W_patches]
        x = x.flatten(2) # [B, embed_dim, num_patches]
        x = x.transpose(1, 2) # [B, num_patches, embed_dim]
        return x

class SpatioTemporalEncoder(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers):
        super().__init__()
        self.patch_embed = TubeletEmbedding(embed_dim=latent_dim, 
                                            patch_size=config.patch_size, 
                                            time_patch_size=config.time_patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, latent_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=latent_dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask_indices=None):
        # x is [B, C, T, H, W]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        if mask_indices is not None:
            # Context Encoder behavior: Keep only visible patches
            # For simplicity in this POC, we just zero out the masked patches
            # In official implementation, they are completely dropped to save compute
            mask = torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
            mask.scatter_(1, mask_indices.unsqueeze(-1), 0)
            x = x * mask
            
        x = self.transformer(x)
        return x

class VJEPAPredictor(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, latent_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=latent_dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, context_encodings, mask_indices):
        B, N, D = context_encodings.shape
        
        # Replace the masked locations with mask tokens + positional embeddings
        pred_input = context_encodings.clone()
        mask_tokens = self.mask_token.expand(B, mask_indices.size(1), D)
        pred_input.scatter_(1, mask_indices.unsqueeze(-1).expand(-1, -1, D), mask_tokens)
        
        pred_input = pred_input + self.pos_embed
        return self.transformer(pred_input)

class VJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_encoder = SpatioTemporalEncoder(config.latent_dim, config.num_heads, config.num_layers)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.predictor = VJEPAPredictor(config.latent_dim, config.num_heads, 1) # Shallow predictor
        
    def update_target_encoder(self):
        with torch.no_grad():
            for ctx_param, tgt_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                tgt_param.data.mul_(config.ema_momentum).add_(ctx_param.data, alpha=1.0 - config.ema_momentum)

model = VJEPA().to(device)
optimizer = optim.AdamW(model.parameters(), lr=config.lr)

# @title 5. Training Loop
def variance_loss(x, target_std=1.0, eps=1e-4):
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(F.relu(target_std - std))

print("Starting V-JEPA Training...")
model.train()

for epoch in range(config.epochs):
    epoch_pred_loss = 0.0
    epoch_var_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
    for video in pbar:
        video = video.to(device) # [B, 1, 8, 28, 28]
        
        # Generate random mask indices (mask 50% of the video tubelets)
        num_mask = num_patches // 2
        rand_indices = torch.rand(config.batch_size, num_patches, device=device).argsort(dim=1)
        mask_indices = rand_indices[:, :num_mask]
        
        optimizer.zero_grad()
        
        # 1. Target Encoder (EMA) processes full video to get targets
        with torch.no_grad():
            full_target_encodings = model.target_encoder(video) # [B, N, D]
            
        # 2. Context Encoder processes video with masked patches hidden
        context_encodings = model.context_encoder(video, mask_indices=mask_indices) # [B, N, D]
        
        # 3. Predictor tries to guess the full sequence
        predicted_encodings = model.predictor(context_encodings, mask_indices)
        
        # 4. Loss: Gather only the masked positions for prediction loss
        B, N, D = full_target_encodings.shape
        target_masked = torch.gather(full_target_encodings, 1, mask_indices.unsqueeze(-1).expand(B, num_mask, D))
        pred_masked = torch.gather(predicted_encodings, 1, mask_indices.unsqueeze(-1).expand(B, num_mask, D))
        
        pred_loss = F.mse_loss(pred_masked, target_masked.detach())
        var_loss = variance_loss(context_encodings.view(-1, D), target_std=config.target_std)
        
        loss = pred_loss + config.variance_weight * var_loss
        
        loss.backward()
        optimizer.step()
        
        model.update_target_encoder()
        
        epoch_pred_loss += pred_loss.item()
        epoch_var_loss += var_loss.item()
        
        pbar.set_postfix({"Pred": f"{pred_loss.item():.4f}", "Var": f"{var_loss.item():.4f}"})

print("Evaluating Variance...")
with torch.no_grad():
    vid = next(iter(train_loader)).to(device)
    enc = model.context_encoder(vid)
    print("Standard Deviation (should be > 0.1):", enc.std(dim=0).mean().item())
