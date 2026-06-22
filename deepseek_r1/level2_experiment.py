# @title 1. Environment Setup
# To run this on Colab:
# !pip install trl peft transformers datasets
# Requires Colab T4 GPU!

import torch
import re
from datasets import Dataset
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    print("WARNING: Required libraries not found. Run: !pip install trl peft transformers datasets")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# @title 2. The Task & Reward Functions
# DeepSeek-R1 proved that rule-based rewards on formatting and correctness
# are all you need to develop reasoning.

def format_reward_func(completions, **kwargs):
    """
    Reward 1.0 if the model wraps its reasoning in <think> tags and provides a final answer.
    """
    rewards = []
    for comp in completions:
        # A simple regex to check if it has <think>...</think> anywhere
        if re.search(r"<think>.*?</think>", comp, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def correctness_reward_func(completions, solution, **kwargs):
    """
    Reward 2.0 if the final answer matches the known solution.
    """
    rewards = []
    for comp, sol in zip(completions, solution):
        # We assume the model puts its final answer at the very end
        if sol.lower() in comp.lower():
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards

# @title 3. Tiny Dataset
# A tiny dataset of simple math/logic problems.
data = [
    {"prompt": "What is 15 + 27?", "solution": "42"},
    {"prompt": "If x + 5 = 12, what is x?", "solution": "7"},
    {"prompt": "What is the capital of France?", "solution": "paris"},
    {"prompt": "What is 8 multiplied by 6?", "solution": "48"}
]

# TRL expects prompts in a specific conversational format
formatted_data = []
for item in data:
    formatted_data.append({
        "prompt": f"Please reason through this step-by-step, wrap your thoughts in <think> and </think> tags, and then provide the final answer.\n\nQuestion: {item['prompt']}",
        "solution": item['solution']
    })

dataset = Dataset.from_list(formatted_data)

# @title 4. Model Setup (Tiny Model + LoRA)
# We use a very small model (0.5B) so it fits in Colab memory while generating G=4 samples.
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

if device == "cuda":
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in bfloat16 to save memory
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    
    # We apply LoRA (Low-Rank Adaptation) so we only train a tiny fraction of the weights.
    # This is crucial for fitting the GRPO group generations in VRAM.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # @title 5. GRPO Training
    print("\nStarting GRPO Training...")
    
    training_args = GRPOConfig(
        output_dir="./grpo_output",
        learning_rate=2e-5,
        num_train_epochs=3,          # Number of passes over the dataset
        per_device_train_batch_size=1, # 1 prompt at a time
        gradient_accumulation_steps=4,
        num_generations=4,           # G=4: Generate 4 answers per prompt to compare!
        max_prompt_length=128,
        max_completion_length=128,   # Keep short for the POC
        logging_steps=1,
        remove_unused_columns=False,
        report_to="none"
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    # @title 6. Inference test
    print("\nTraining Complete! Let's test the model's new reasoning behavior:")
    test_prompt = "Please reason through this step-by-step, wrap your thoughts in <think> and </think> tags, and then provide the final answer.\n\nQuestion: What is 10 + 20?"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    print("\nGenerated Output:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
else:
    print("WARNING: This experiment requires a GPU to load the HuggingFace model. Please run this on Colab T4 GPU.")
