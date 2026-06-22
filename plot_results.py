import matplotlib.pyplot as plt
import os

os.makedirs('assets', exist_ok=True)

# 1. I-JEPA Loss Curve
ijepa_epochs = list(range(1, 11))
ijepa_loss = [0.1360, 0.0936, 0.0832, 0.0777, 0.0779, 0.0753, 0.0765, 0.0776, 0.0763, 0.0753]

plt.figure(figsize=(8, 5))
plt.plot(ijepa_epochs, ijepa_loss, marker='o', color='#1f77b4', linewidth=2)
plt.title('I-JEPA Training Loss (FashionMNIST)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Smooth L1 Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(ijepa_epochs)
plt.tight_layout()
plt.savefig('assets/ijepa_loss.png', dpi=300)
plt.close()

# 2. Mamba Loss Curve
mamba_epochs = list(range(1, 11))
mamba_loss = [2.7079, 2.4788, 2.4681, 2.4648, 2.4642, 2.4618, 2.4610, 2.4596, 2.4581, 2.4609]

plt.figure(figsize=(8, 5))
plt.plot(mamba_epochs, mamba_loss, marker='s', color='#2ca02c', linewidth=2)
plt.title('Mamba SSM Training Loss (Text Generation)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cross Entropy Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(mamba_epochs)
plt.tight_layout()
plt.savefig('assets/mamba_loss.png', dpi=300)
plt.close()

# 3. V-JEPA 2 Loss Curve
vjepa_epochs = [100, 200, 300, 400, 500]
vjepa_loss = [0.3124, 0.2726, 0.2462, 0.2124, 0.1954]

plt.figure(figsize=(8, 5))
plt.plot(vjepa_epochs, vjepa_loss, marker='D', color='#ff7f0e', linewidth=2)
plt.title('V-JEPA 2 Action-Conditioned Predictor Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss (Latent Space)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(vjepa_epochs)
plt.tight_layout()
plt.savefig('assets/vjepa2_loss.png', dpi=300)
plt.close()

# 4. Test-Time Compute Comparison
strategies = ['Greedy Decoding\n(1x Compute)', 'Best-of-N\n(16x Compute)']
scores = [0.0, 1.0]

plt.figure(figsize=(8, 5))
bars = plt.bar(strategies, scores, color=['#d62728', '#9467bd'])
plt.title('Scaling Test-Time Compute (Math Reasoning)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy Score', fontsize=12)
plt.ylim(0, 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('assets/test_time_compute.png', dpi=300)
plt.close()

print("Graphs successfully generated in assets/ folder.")
