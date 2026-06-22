# @title 1. Environment Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import requests
import math
import os
from tqdm import tqdm

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# @title 2. Dataset: Tiny Shakespeare
def get_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("input.txt"):
        print("Downloading Tiny Shakespeare...")
        r = requests.get(url)
        with open("input.txt", "w") as f:
            f.write(r.text)
    with open("input.txt", "r") as f:
        text = f.read()
    return text

text = get_shakespeare()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# @title 3. Configuration
class Config:
    batch_size = 32
    block_size = 64  # sequence length
    vocab_size = vocab_size
    d_model = 64
    d_state = 16
    n_layers = 2
    epochs = 5
    lr = 3e-3

config = Config()

def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - config.block_size, (config.batch_size,))
    x = torch.stack([data_split[i : i + config.block_size] for i in ix])
    y = torch.stack([data_split[i + 1 : i + config.block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# @title 4. Minimal Pure PyTorch Mamba Block (Sequential)
# Note: This is an unoptimized, pure-PyTorch implementation of the Selective SSM.
# It runs sequentially like an RNN. The real Mamba uses parallel scan in CUDA.

class MinimalMambaBlock(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Projections for x -> Delta, B, C
        self.proj_delta = nn.Linear(d_model, d_model)
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_model, d_state)
        
        # Base continuous A matrix (using a simple random initialization for POC)
        # In the real paper, this is initialized as the HiPPO matrix and kept constant or learned slowly.
        A = torch.randn(d_model, d_state) / math.sqrt(d_state)
        self.A_log = nn.Parameter(torch.log(torch.abs(A))) # Parameterize in log space for stability
        
        # Out projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        
        # Compute selectivity params for all tokens in parallel
        # x: [B, T, C]
        delta = F.softplus(self.proj_delta(x)) # [B, T, C] - step size must be positive
        B_mat = self.proj_B(x) # [B, T, d_state]
        C_mat = self.proj_C(x) # [B, T, d_state]
        
        A = -torch.exp(self.A_log) # [C, d_state] - negative for stable continuous system
        
        # Recurrent scan (Slow in PyTorch, Fast in CUDA)
        y = torch.zeros_like(x)
        h = torch.zeros(B, C, self.d_state, device=x.device) # [B, C, d_state]
        
        for t in range(T):
            x_t = x[:, t, :] # [B, C]
            delta_t = delta[:, t, :] # [B, C]
            B_t = B_mat[:, t, :] # [B, d_state]
            C_t = C_mat[:, t, :] # [B, d_state]
            
            # Discretize continuous A and B (Zero-order hold)
            # A_bar = exp(Delta * A)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A) # [B, C, d_state]
            # B_bar = (Delta * A)^-1 * (A_bar - I) * B
            # Simplified approximation used in Mamba for numerical stability: B_bar = Delta * B
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1) # [B, C, d_state]
            
            # State Update: h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar * h + B_bar * x_t.unsqueeze(-1) # [B, C, d_state]
            
            # Output generation: y_t = C_t * h_t
            # h is [B, C, d_state], C_t is [B, d_state]. We need dot product along d_state.
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) # [B, C]
            y[:, t, :] = y_t
            
        return self.out_proj(y)

class TinyMambaLM(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MinimalMambaBlock(d_model, d_state) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x, targets=None):
        x = self.embedding(x)
        for layer in self.layers:
            # Simple residual connection
            x = x + layer(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

model = TinyMambaLM(config.vocab_size, config.d_model, config.d_state, config.n_layers).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config.lr)

# @title 5. Training Loop
print(f"Starting Training on {device} (This might be slow due to sequential loop in Python)")
model.train()

# 5 epochs, let's say 100 iterations per epoch for the POC
iters_per_epoch = 100
for epoch in range(config.epochs):
    total_loss = 0
    pbar = tqdm(range(iters_per_epoch), desc=f"Epoch {epoch+1}/{config.epochs}")
    for _ in pbar:
        x, y = get_batch("train")
        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    avg_loss = total_loss / iters_per_epoch
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# Generate text
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = []
print("\nGenerating text:")
with torch.no_grad():
    for _ in range(200):
        logits, _ = model(context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, num_samples=1)
        generated.append(next_char.item())
        context = torch.cat((context, next_char), dim=1)

print(decode(generated))
