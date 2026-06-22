# Scaling Test-Time Compute Experiment Results

## Level 1 Proof of Concept (CPU)

**Objective:** Mathematically demonstrate the "pass@k" scaling law, showing how Best-of-N Search dramatically improves effective accuracy even for weak base models.

**Execution Log:**
*Paste your output here.*

**Observations:**
*Did you notice how quickly the theoretical accuracy scales up as N increases?*

---

## Level 2 Experiment (T4 GPU)

**Objective:** Test an open-source model (Qwen 0.5B) on a tricky logic problem (the Bat & Ball problem) to compare Greedy Decoding vs. Best-of-N vs. Sequential Revision.

**Execution Log:**
*Paste your output here.*

**Observations:**
*Did the baseline greedy decoding fail? Which test-time scaling strategy worked best for the model to "think" its way to the correct answer (5 cents)?*
