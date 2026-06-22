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

4. **[V-JEPA 2 (Robotics)](v_jepa_2/)** ✅ *Completed*
   > Zero-shot robotic manipulation using world models trained on web video.

### Part 2: Scaling, Reasoning, and Efficiency
5. **[Mamba & Mamba-2 (SSMs)](mamba_ssm/)** ✅ *Completed*
   > Replacing attention with selective state space models for linear-time scaling.

6. **[DeepSeek-R1](deepseek_r1/)** ✅ *Completed*
   > Reinforcement learning for reasoning without supervised fine-tuning.

7. **[Scaling LLM Test-Time Compute](test_time_compute/)** ✅ *Completed*
   > Best-of-N sampling and inference-time search to beat larger models.

8. **[Constitutional AI](constitutional_ai/)** ✅ *Completed*
   > Scalable oversight and self-revision for harmlessness.

9. **[Large Concept Models (LCMs)](large_concept_models/)** ✅ *Completed*
   > Autoregressive generation over semantic sentence embeddings.

---

## 📊 Architectural Graphs

Here are the core architectures of the papers explored in this repository.

### 1. The JEPA Architecture (LeCun 2022 / I-JEPA / V-JEPA)
```mermaid
graph TD
    X[Context x] --> Enc[Encoder]
    Y[Target y] --> TargetEnc[Target Encoder]
    Z[Latent Variable z] --> Pred[Predictor]
    
    Enc --> sx[Representation sx]
    TargetEnc --> sy[Representation sy]
    
    sx --> Pred
    Pred --> sy_pred[Predicted sy]
    
    sy --> Loss((Loss))
    sy_pred --> Loss
    
    style Enc fill:#1f77b4,color:#fff
    style TargetEnc fill:#1f77b4,color:#fff
    style Pred fill:#ff7f0e,color:#fff
    style Loss fill:#d62728,color:#fff
```
*Caption: The core Joint-Embedding Predictive Architecture. Unlike standard networks that predict pixels (y), JEPA predicts the abstract representation of y (sy) in latent space.*

### 2. DeepSeek-R1 GRPO (Group Relative Policy Optimization)
```mermaid
graph TD
    Prompt[Prompt] --> Base[Base LLM]
    Base --> O1[Output 1]
    Base --> O2[Output 2]
    Base --> On[Output N]
    
    O1 --> R1[Rule-based Reward]
    O2 --> R2[Rule-based Reward]
    On --> Rn[Rule-based Reward]
    
    R1 --> Adv[Normalize to Advantages]
    R2 --> Adv
    Rn --> Adv
    
    Adv --> PPO[PPO Clipping Objective]
    KL[KL Divergence Penalty] --> PPO
    
    PPO --> Update[Update Base LLM Weights]
    
    style Base fill:#2ca02c,color:#fff
    style R1 fill:#9467bd,color:#fff
    style PPO fill:#d62728,color:#fff
```
*Caption: DeepSeek-R1's GRPO removes the need for an expensive Critic Model. It generates N outputs, scores them using logic/math rules, normalizes the scores, and updates the policy.*

### 3. Scaling Test-Time Compute
```mermaid
graph LR
    P[Prompt] --> LLM1[LLM]
    LLM1 --> O1[Draft 1]
    O1 --> Crit[Critique Draft 1]
    Crit --> LLM2[LLM]
    LLM2 --> O2[Draft 2]
    O2 --> Crit2[Critique Draft 2]
    Crit2 --> LLM3[LLM]
    LLM3 --> Final[Final Answer]
    
    style LLM1 fill:#17becf,color:#fff
    style Crit fill:#bcbd22,color:#fff
```
*Caption: Sequential Revision (Iterative Compute). Instead of scaling up the model parameters, we scale the time the model spends thinking and revising during inference.*

### 4. Large Concept Models (LCMs)
```mermaid
graph LR
    S1[Sentence 1 Text] --> SONAR1[SONAR Encoder]
    SONAR1 --> V1[Concept Vector 1]
    
    V1 --> LCM[Continuous Space LCM]
    LCM --> V2[Predicted Concept Vector 2]
    
    V2 --> SONAR2[SONAR Decoder]
    SONAR2 --> S2[Sentence 2 Text]
    
    style SONAR1 fill:#8c564b,color:#fff
    style LCM fill:#e377c2,color:#fff
    style SONAR2 fill:#8c564b,color:#fff
```
*Caption: LCMs detach reasoning from language. A sentence is encoded into a high-dimensional concept, the LCM predicts the next concept, and the decoder translates it back to text.*

---

## 🛠️ Implementation Strategy
## 📈 Experimental Results (Colab Runs)

Here are the actual training and evaluation graphs generated from our Colab T4 experiments!

### 1. I-JEPA Representation Learning
![I-JEPA Loss](assets/ijepa_loss.png)
*Caption: I-JEPA successfully learning to predict contextual patches in latent space, dropping the Smooth L1 Loss from 0.13 to 0.075 on FashionMNIST.*

### 2. Mamba (State Space Models)
![Mamba Loss](assets/mamba_loss.png)
*Caption: Mamba demonstrating stable autoregressive text generation loss reduction over 10 epochs.*

### 3. V-JEPA 2 (Robotic Planning)
![V-JEPA 2 Loss](assets/vjepa2_loss.png)
*Caption: V-JEPA 2 learning an Action-Conditioned World Model, enabling zero-shot Model Predictive Control (MPC).*

### 4. Scaling Test-Time Compute
![Test-Time Compute](assets/test_time_compute.png)
*Caption: Scaling compute at inference time. The Best-of-N search strategy perfectly solved a math reasoning problem that Greedy Decoding failed on, showing a 100% accuracy boost just by thinking longer.*

### 5. V-JEPA Representation Learning
![V-JEPA Loss](assets/vjepa_loss.png)
*Caption: V-JEPA successfully learning prediction over spatiotemporal video cubes, showing a steady decrease in prediction loss over 10 epochs.*

### 6. Large Concept Models (LCM)
![LCM Loss](assets/lcm_loss.png)
*Caption: LCM achieving 0.00 loss on continuous trajectories and successfully generating concept predictions in the latent space.*

### 7. DeepSeek-R1 (GRPO)
![DeepSeek-R1 Reward](assets/deepseek_r1_reward.png)
*Caption: DeepSeek-R1 optimizing rule-based reasoning rewards (GRPO), rapidly pushing the model reward to ~2.0.*

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
