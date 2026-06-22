# V-JEPA 2 (Robotic Planning) Experiment Results

## Level 1 Proof of Concept (CPU)

**Objective:** Understand how Model Predictive Control (MPC) works when the "Model" is an abstract latent predictor rather than a physics engine.

**Execution Log:**
*Paste your output here.*

**Observations:**
*Notice how the action evaluation happens entirely inside the latent space—the code never applies the proposed action to the `current_state` variable during the planning loop!*

---

## Level 2 Experiment (T4 GPU)

**Objective:** Train an Action-Conditioned Encoder and Predictor on a dataset of simulated object trajectories, then use it to perform zero-shot Model Predictive Control to reach a target.

**Execution Log:**
*Paste your output here.*

**Observations:**
*Did the planner find the optimal action vector near `[8.0, 8.0]`? This simple experiment is the exact mechanism Meta FAIR used to make robotic arms fold clothes after watching YouTube videos!*
