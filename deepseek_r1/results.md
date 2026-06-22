# DeepSeek-R1 Experiment Results

## Level 1 Proof of Concept (CPU, Pure PyTorch)

**Objective:** Verify the mathematical foundation of Group Relative Policy Optimization (GRPO), specifically standardizing rewards as advantages and computing the clipped surrogate loss with KL divergence penalty.

**Execution Log:**
*Paste your output here.*

**Observations:**
*Notice how the gradient pushes the probabilities: did the output with the highest reward get the strongest negative gradient (which corresponds to an "increase probability" signal)?*

---

## Level 2 Experiment (T4 GPU, TRL GRPOTrainer)

**Objective:** Train a tiny language model (Qwen 0.5B) to develop reasoning chains (using `<think>` tags) and output correct answers using *only* reinforcement learning and rule-based rewards.

**Execution Log:**
*Paste your training output here.*

**Observations:**
*Did the model learn to reliably output `<think>` tags by the end of training? Paste its answer to the test prompt!*
