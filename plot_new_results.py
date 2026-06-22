import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('assets', exist_ok=True)

# 1. V-JEPA Loss Curve
vjepa_epochs = list(range(1, 11))
vjepa_loss = [0.1030, 0.0834, 0.1002, 0.1435, 0.1755, 0.1774, 0.1753, 0.1799, 0.1855, 0.1769]

plt.figure(figsize=(8, 5))
plt.plot(vjepa_epochs, vjepa_loss, marker='o', color='#1f77b4', linewidth=2)
plt.title('V-JEPA Level 2 Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Prediction Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(vjepa_epochs)
plt.tight_layout()
plt.savefig('assets/vjepa_loss.png', dpi=300)
plt.close()

# 2. Large Concept Models (LCM) Loss
lcm_epochs = [100, 200, 300]
lcm_loss = [0.0000, 0.0000, 0.0000]

plt.figure(figsize=(8, 5))
plt.plot(lcm_epochs, lcm_loss, marker='s', color='#2ca02c', linewidth=2)
plt.title('Large Concept Models (LCM) Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.ylim(-0.1, 0.1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(lcm_epochs)
plt.tight_layout()
plt.savefig('assets/lcm_loss.png', dpi=300)
plt.close()

# 3. DeepSeek R1 GRPO Reward & Loss
r1_epochs = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
r1_reward = [1.5, 2.0, 2.25, 2.0, 2.0, 2.25, 2.0, 1.75, 2.0, 2.0, 1.75, 2.0]

plt.figure(figsize=(8, 5))
plt.plot(r1_epochs, r1_reward, marker='D', color='#ff7f0e', linewidth=2)
plt.title('DeepSeek-R1 GRPO Training Reward', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.ylim(0, 2.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(r1_epochs)
plt.tight_layout()
plt.savefig('assets/deepseek_r1_reward.png', dpi=300)
plt.close()

print("New graphs successfully generated in assets/ folder.")
