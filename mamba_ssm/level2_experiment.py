# @title 1. Environment Setup
# To run this on Colab:
# !pip install causal-conv1d>=1.2.0
# !pip install mamba-ssm
# Requires Colab GPU Runtime!

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import requests
import os
from tqdm import tqdm

try:
    from mamba_ssm import Mamba
except ImportError:
    print("WARNING: mamba-ssm is not installed. Please run '!pip install causal-conv1d mamba-ssm' in Colab.")
    # Fallback dummy class so the file still parses
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state, d_conv, expand):
            super().__init__()
            self.net = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.net(x)

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
    batch_size = 64
    block_size = 256  # Mamba can handle much larger contexts easily!
    vocab_size = vocab_size
    d_model = 128
    d_state = 16
    d_conv = 4
    expand = 2
    n_layers = 4
    epochs = 10
    lr = 1e-3

config = Config()

def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - config.block_size, (config.batch_size,))
    x = torch.stack([data_split[i : i + config.block_size] for i in ix])
    y = torch.stack([data_split[i + 1 : i + config.block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# @title 4. Mamba Language Model Architecture
class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, d_conv, expand, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # We wrap the official Mamba block in a residual connection + layer norm
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mixer': Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                ),
                'norm': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x, targets=None):
        x = self.embedding(x)
        
        for layer in self.layers:
            residual = x
            x = layer['norm'](x)
            x = layer['mixer'](x)
            x = x + residual
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

if device == "cuda":
    model = MambaLM(
        config.vocab_size, 
        config.d_model, 
        config.d_state, 
        config.d_conv, 
        config.expand, 
        config.n_layers
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # @title 5. Training Loop
    print(f"Starting Training on {device} using official mamba-ssm")
    model.train()

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
        for _ in range(500):
            logits, _ = model(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1)
            generated.append(next_char.item())
            
            # Fast inference: Mamba handles long contexts linearly!
            # (Note: proper caching is needed for true O(1) inference speed, 
            # but this demonstrates sequence modeling ability)
            context = torch.cat((context, next_char), dim=1)

    print(decode(generated))
else:
    print("WARNING: This experiment requires a GPU to run `mamba-ssm`. Please run this on Colab T4 GPU.")
