# 🧠 AI Research Implementation & Study Repository

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Welcome to my personal AI research and implementation repository. This project serves as a rigorous deep dive into the most foundational and high-impact papers in Artificial Intelligence—ranging from Joint-Embedding Predictive Architectures (JEPA) and world models, to reinforcement learning for reasoning, and State Space Models (SSMs).

Our goal is not just to read, but to **implement constraint-aware versions** of every paper that can run on free-tier compute (Google Colab T4 GPUs, GitHub Codespaces CPUs).

---

## 🗺️ Learning Path & Roadmap

The implementation plan follows a curated track, building from foundational theories to state-of-the-art applied models.

### Part 1: The JEPA Family (World Models & Representations)
1. **[LeCun 2022 Position Paper](lecun_2022_jepa/)** ✅ *Completed*
   > *A Path Towards Autonomous Machine Intelligence.* Establishing the theoretical groundwork for predicting abstract representations.
   
2. **[I-JEPA](i_jepa/)** ✅ *Completed*
   > *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.*

3. **[MC-JEPA & V-JEPA](v_jepa/)** ✅ *Completed*
   > Extending representation learning to motion and full video spatiotemporal cubes.

4. **V-JEPA 2 (Robotics)** ⏳ *Planned*
   > Zero-shot robotic manipulation using world models trained on web video.

### Part 2: Scaling, Reasoning, and Efficiency
5. **[Mamba & Mamba-2 (SSMs)](mamba_ssm/)** ✅ *Completed*
   > Replacing attention with selective state space models for linear-time scaling.

6. **[DeepSeek-R1](deepseek_r1/)** ✅ *Completed*
   > Reinforcement learning for reasoning without supervised fine-tuning.

7. **Scaling LLM Test-Time Compute** 🔄 *In Progress*
   > Best-of-N sampling and inference-time search to beat larger models.

8. **Constitutional AI** ⏳ *Planned*
   > Scalable oversight and self-revision for harmlessness.

9. **Large Concept Models (LCMs)** ⏳ *Planned*
   > Autoregressive generation over semantic sentence embeddings.

---

## 🛠️ Implementation Strategy

For each paper, you will find:
- `study_notes.md`: A plain-English explanation, visual mental models, prerequisites, and constraint-aware implementation plans.
- `level1_poc.py`: A tiny toy version that proves the core idea works on a CPU in under 5 minutes.
- `level2_experiment.py`: A scaled-down but meaningful version designed to fit into a Google Colab T4 (12GB VRAM).
- `results.md`: Training logs, visualisations, and insights gained during execution.

---

## 🚀 Getting Started

To explore the implementations locally or on Colab:
```bash
git clone https://github.com/Paramveersingh-S/Research.git
cd Research
```
Open any of the `level2_experiment.py` files and paste them into a Google Colab notebook to begin experimenting!
