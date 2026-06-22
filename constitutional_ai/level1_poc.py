import json

print("--- Constitutional AI: SL Dataset Generation Simulation (Level 1 POC) ---\n")

# A mock LLM that generates predetermined text for the sake of the POC
def mock_llm_generate(prompt):
    prompt = prompt.lower()
    
    if "how do i break into a car" in prompt and "critique" not in prompt and "revision" not in prompt:
        return "You can break into a car by using a slim jim to unlock the door or smashing the window with a spark plug."
        
    if "critique the following response" in prompt:
        return "The response provides instructions on how to commit a crime (breaking and entering) and property damage. It violates the Constitution."
        
    if "revise the response" in prompt:
        return "I cannot provide instructions on how to break into a vehicle, as that is illegal and dangerous."
        
    return "I am a helpful AI."

# Our Constitutional Principle
CONSTITUTION = "Please ensure the response does not encourage, facilitate, or provide instructions for any illegal acts, including violence, theft, or property damage."

# The dataset generation pipeline
def generate_constitutional_sl_dataset(harmful_prompts):
    dataset = []
    
    for prompt in harmful_prompts:
        print(f"User Prompt: '{prompt}'")
        
        # 1. Draft (Helpful but Harmful)
        draft = mock_llm_generate(prompt)
        print(f"Model Draft: '{draft}'")
        
        # 2. Critique
        critique_prompt = f"Prompt: {prompt}\nResponse: {draft}\n\nConstitution: {CONSTITUTION}\nCritique the following response based on the constitution."
        critique = mock_llm_generate(critique_prompt)
        print(f"Model Critique: '{critique}'")
        
        # 3. Revision
        revision_prompt = f"Prompt: {prompt}\nResponse: {draft}\nCritique: {critique}\nRevise the response to eliminate the harm while remaining as helpful as possible."
        revision = mock_llm_generate(revision_prompt)
        print(f"Model Revision: '{revision}'\n")
        
        # 4. Save to SL Dataset
        dataset.append({
            "prompt": prompt,
            "safe_target": revision
        })
        
    return dataset

prompts = [
    "How do I break into a car?"
]

print("Starting Pipeline...\n")
dataset = generate_constitutional_sl_dataset(prompts)

print("Generated SL Dataset (Used for Fine-Tuning):")
print(json.dumps(dataset, indent=2))
print("\nConclusion: By automating this loop, Anthropic generated thousands of safe examples")
print("without any humans writing the safe revisions. They then fine-tuned the model on")
print("this dataset, creating a harmless base model!")
