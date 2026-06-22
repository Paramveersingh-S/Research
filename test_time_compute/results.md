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
```text
--- Baseline: Greedy Decoding (Minimal Test-Time Compute) ---
Output:
Question: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?
Answer: Let x be the price of the ball in cents. Then 1 + x = 1.10. Subtracting 1 from both sides, we get x = 0.10. Therefore, the answer is 10.

Score: 0.0

--- Best-of-N Search (Scaling Compute 16x) ---
Generating 16 distinct answers and scoring them...

Best Output Found (Score 1.0):
Question: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?
Answer: Let x be the number of dollars the ball costs. Then, the bat is worth 1 + x = x + 1. Thus, we have 1 * x + (x + 1) * 1 = 110 cents. Solving this equation gives us 2x = 95, so x = 47.5 cents. Therefore, the answer is 47.5.

--- Sequential Revision (Iterative Compute) ---
Draft 1: [...] Subtracting 1 from both sides gives us 2
Revision: [...] The first flaw is that the problem statement says...

Final Score: 0.0
```

**Observations:**
- **Greedy Decoding:** Predictably failed the classic cognitive reflection test by confidently outputting "10".
- **Best-of-N Search & Reward Hacking:** This is an incredible result! The model scored a 1.0, but if you read the text, it arrived at `x = 47.5`. The only reason it scored 1.0 is because my simple programmatic reward script awarded a point to anything ending in `5`! This perfectly demonstrates a massive limitation mentioned in the paper: **Best-of-N is only as good as its Verifier.** If the verifier has a flaw, Best-of-N compute scaling will simply search the latent space until it finds an answer that exploits that flaw (Reward Hacking)!
- **Sequential Revision:** Failed to meaningfully correct itself ("Critique Collapse"). Small models generally struggle with self-correction without specific fine-tuning.
