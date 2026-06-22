# Mamba (SSM) Experiment Results

## Level 1 Proof of Concept (CPU, Pure PyTorch)

**Objective:** Verify the mathematical foundation of selective state space models (discretization via Zero-Order Hold and selective gating via linear projections) using a minimal, pure PyTorch recurrent implementation.

**Execution Log:**
*Paste your training output here.*

**Observations:**
*Did the model train properly despite running sequentially in pure PyTorch? Note the speed limitations compared to the parallel scan.*

---

## Level 2 Experiment (T4 GPU, Official Mamba Package)

**Objective:** Train a language model on Tiny Shakespeare using the official hardware-aware `mamba-ssm` package.

**Execution Log:**
*Paste your training output here.*

**Observations:**
*Note the sequence length vs VRAM usage, and training speed. Did the loss decrease steadily? Paste a sample of the generated Shakespearean text!*
