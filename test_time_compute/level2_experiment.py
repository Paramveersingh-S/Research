# @title 1. Environment Setup
# To run this on Colab:
# !pip install transformers accelerate
# Requires Colab T4 GPU!

import torch
import re
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("WARNING: transformers not installed. Run: !pip install transformers")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# @title 2. Load Tiny Model
# We use a very small model so generation is fast enough for testing inference scaling.
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

if device == "cuda":
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    # A tricky math problem that small models often get wrong on the first try
    PROBLEM = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?"
    CORRECT_ANSWER = "5" # 5 cents

    def check_answer(text):
        """A simple rule-based Reward Model (Verifier)."""
        # We look for "5" or "5 cents" in the final text.
        # Often models confidently say "10 cents".
        if " 5 " in text or "5 cents" in text.lower() or text.strip().endswith("5"):
            return 1.0
        return 0.0

    # @title 3. Baseline: Greedy Decoding (N=1)
    print("\n--- Baseline: Greedy Decoding (Minimal Test-Time Compute) ---")
    inputs = tokenizer([f"Question: {PROBLEM}\nAnswer:"], return_tensors="pt").to(device)
    
    # Greedy: temperature=0
    greedy_output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    print(f"Output:\n{greedy_text}\n")
    print(f"Score: {check_answer(greedy_text)}")


    # @title 4. Best-of-N Scaling (Parallel Test-Time Compute)
    print("\n--- Best-of-N Search (Scaling Compute 16x) ---")
    N = 16
    print(f"Generating {N} distinct answers and scoring them...")
    
    # We use high temperature to encourage diverse reasoning paths
    bon_outputs = model.generate(
        **inputs, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.9,
        top_p=0.9,
        num_return_sequences=N
    )
    
    best_text = None
    best_score = -1.0
    
    for i, out in enumerate(bon_outputs):
        text = tokenizer.decode(out, skip_special_tokens=True)
        score = check_answer(text)
        if score > best_score:
            best_score = score
            best_text = text
            
    print(f"\nBest Output Found (Score {best_score}):\n{best_text}\n")


    # @title 5. Sequential Revision (Iterative Test-Time Compute)
    print("\n--- Sequential Revision (Iterative Compute) ---")
    # Instead of randomly sampling, we force the model to critique its own greedy answer
    
    draft_prompt = f"Question: {PROBLEM}\nAnswer: Let's think step by step."
    draft_inputs = tokenizer([draft_prompt], return_tensors="pt").to(device)
    draft_output = model.generate(**draft_inputs, max_new_tokens=100, do_sample=False)
    draft_text = tokenizer.decode(draft_output[0], skip_special_tokens=True)
    
    print(f"Draft 1:\n{draft_text}\n")
    
    # Now append a critique request
    critique_prompt = f"{draft_text}\n\nCritique: Read the draft above carefully. Find exactly one flaw in the math logic, and output a corrected Final Answer."
    critique_inputs = tokenizer([critique_prompt], return_tensors="pt").to(device)
    
    revision_output = model.generate(**critique_inputs, max_new_tokens=100, do_sample=False)
    revision_text = tokenizer.decode(revision_output[0], skip_special_tokens=True)
    
    # Only print the new part
    new_text = revision_text[len(draft_text):]
    print(f"Revision:\n{new_text}\n")
    print(f"Final Score: {check_answer(revision_text)}")

else:
    print("WARNING: This experiment requires a GPU. Please run on Colab T4.")
