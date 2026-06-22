import random

print("--- Scaling LLM Test-Time Compute (Level 1 POC) ---")
print("Simulating how 'Best-of-N' search beats a single greedy generation.\n")

def simulate_best_of_n(base_probability, num_samples, num_trials=10000):
    """
    Simulates the 'pass@k' (or Best-of-N) scaling law.
    If a model has a `base_probability` of getting the right answer on any single generation,
    what is the probability that AT LEAST ONE of the `num_samples` is correct?
    
    This assumes we have a perfect Reward Model (Verifier) that can always identify the correct 
    answer if the generator manages to produce it.
    """
    successes = 0
    for _ in range(num_trials):
        # Generate N samples
        samples = [random.random() < base_probability for _ in range(num_samples)]
        
        # Did at least one succeed?
        if any(samples):
            successes += 1
            
    empirical_prob = successes / num_trials
    
    # Mathematical calculation: 1 - P(failing all N times)
    theoretical_prob = 1.0 - ((1.0 - base_probability) ** num_samples)
    
    return empirical_prob, theoretical_prob

# Scenario: The problem is hard. The LLM only gets it right 10% of the time.
# A standard user just asks the LLM once (Greedy decoding).
base_prob = 0.10

print(f"Base Model Accuracy (Greedy / N=1): {base_prob*100:.1f}%\n")

print("Scaling Test-Time Compute (Sampling N times and picking the best):")
print("-" * 65)
print(f"{'N (Compute Budget)':<20} | {'Empirical Accuracy':<20} | {'Theoretical Accuracy':<20}")
print("-" * 65)

test_budgets = [1, 2, 5, 10, 50, 100]

for n in test_budgets:
    emp, theo = simulate_best_of_n(base_prob, n)
    print(f"{n:<20} | {emp*100:>18.1f}% | {theo*100:>19.1f}%")

print("-" * 65)
print("\nConclusion:")
print("Even if an LLM is terrible at a task (10% accuracy), if you spend 100x the inference compute")
print("(N=100) and have a verifier to pick the winner, its effective accuracy jumps to >99%!")
print("This is the core mathematical engine behind OpenAI's o1 and DeepSeek-R1 test-time search.")
