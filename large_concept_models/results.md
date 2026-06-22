# Large Concept Models Experiment Results

## Level 1 Proof of Concept (CPU)

**Objective:** Prove that sequence prediction can happen in continuous space using PyTorch's `MSELoss`, instead of standard token-space using `CrossEntropyLoss`.

**Execution Log:**
*Paste your output here.*

**Observations:**
*Notice how the model predicted vectors that were mathematically very close to the true target vectors, effectively generating the sequence `0 -> 1 -> 2` purely via spatial reasoning!*

---

## Level 2 Experiment (T4 GPU)

**Objective:** Use a Sentence Transformer to embed a story into Concept Vectors, train a Multi-Layer Perceptron (the "LCM") to predict the trajectory of the story in latent space, and decode it by finding the nearest neighbor.

**Execution Log:**
*Paste your output here.*

**Observations:**
*Did the LCM successfully predict the concept of "Lightning flashed across the horizon" when prompted with "The sky grew dark"? This shows how AI can plan ideas hierarchically before translating them back into language!*
