#!/usr/bin/env python3
# repairity_eval.py - Evaluation script for REPAIRITY models

import os
import json
import torch
import random
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Test prompts to evaluate the model's reasoning abilities
TEST_PROMPTS = [
    "Write a function to find the longest common subsequence of two strings",
    "Implement a solution to the N-Queens problem",
    "Create a function to detect if a linked list has a cycle",
    "Write a program to find all prime numbers up to n using the Sieve of Eratosthenes",
    "Implement a binary search tree with insert and delete operations"
]

# Use some BigCodeBench examples that weren't used in training
def get_test_examples(num_examples=5, seed=42, claude_test_path=None):
    """Get test examples that weren't used in training."""
    
    # First try to use Claude test data if available
    if claude_test_path and os.path.exists(claude_test_path):
        print(f"Loading Claude test data from {claude_test_path}")
        with open(claude_test_path, 'r') as f:
            claude_test_data = json.load(f)
            
        # If we have more examples than requested, sample randomly
        if len(claude_test_data) > num_examples:
            random.seed(seed)
            test_samples = random.sample(claude_test_data, num_examples)
        else:
            test_samples = claude_test_data
            
        # Format the examples for testing
        formatted_examples = []
        for item in test_samples:
            formatted_examples.append({
                "id": item.get("task_id", "unknown"),
                "prompt": item.get("task", ""),
                "reference_solution": item.get("reference_solution", "")
            })
            
        return formatted_examples
    
    # Load examples from BigCodeBench test set
    # Load the dataset
    dataset = load_dataset("bigcode/bigcodebench")
    version_data = dataset["v0.1.4"]
    
    # Check if we have saved train/test indices
    if os.path.exists("./data/train_test_indices.json"):
        with open("./data/train_test_indices.json", "r") as f:
            indices = json.load(f)
            train_indices = set(indices.get("train_indices", []))
            test_indices = list(indices.get("test_indices", []))
    else:
        # If no indices file, split the data randomly
        print("No train/test indices found, creating new split")
        all_indices = list(range(len(version_data)))
        random.seed(seed)
        random.shuffle(all_indices)
        split_idx = int(len(all_indices) * 0.9)  # 90% train, 10% test
        train_indices = set(all_indices[:split_idx])
        test_indices = all_indices[split_idx:]
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Select random test examples
    selected_indices = random.sample(test_indices, min(num_examples, len(test_indices)))
    test_examples = [version_data[i] for i in selected_indices]
    
    # Format the examples for testing
    return [format_test_input(ex) for ex in test_examples]

def format_test_input(example):
    """Format a test example for input to the model."""
    # Extract task details
    task_id = example.get("task_id", "")
    instruct_prompt = example.get("instruct_prompt", "")
    code_prompt = example.get("code_prompt", "")
    complete_prompt = example.get("complete_prompt", "")
    
    # Choose the best prompt available
    if instruct_prompt:
        instruction = instruct_prompt
    elif complete_prompt:
        instruction = complete_prompt
    elif code_prompt:
        instruction = f"Complete the following code:\n\n{code_prompt}"
    else:
        instruction = f"Write code to solve the following task (ID: {task_id})."
    
    return {
        "id": task_id,
        "prompt": instruction,
        "reference_solution": example.get("canonical_solution", "")
    }

def load_model(model_path, device="cuda", quantization=True):
    """Load a model for evaluation."""
    print(f"Loading model from {model_path}")
    
    # Determine if it's a LoRA model
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_lora:
        print("Detected LoRA model")
        # Get base model name from config
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            config = json.load(f)
            base_model_name = config.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-7B-Instruct")
        
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        # Load base model with quantization if needed
        if quantization and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Load PEFT model
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print("Loading regular model")
        # Load tokenizer and model directly
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if quantization and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, system_prompt="You are a skilled programmer. Provide complete, efficient, and correct code solutions with detailed reasoning."):
    """Generate a response from the model."""
    # Format as a chat
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(chat, tokenize=False)
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,
            top_p=0.95,
            do_sample=True
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    assistant_response = response.split("ASSISTANT: ")[-1].strip()
    
    return assistant_response

def evaluate_model(model_path, num_examples=5, claude_test_path=None):
    """Evaluate a model on test examples."""
    # Load model and tokenizer
    model, tokenizer = load_model(model_path)
    
    # Get test examples - first try Claude test data, then fall back to BigCodeBench
    formatted_examples = get_test_examples(num_examples, claude_test_path=claude_test_path)
    
    # Add some standard programming challenges
    for prompt in TEST_PROMPTS[:num_examples]:
        formatted_examples.append({
            "id": f"custom_{prompt.split()[1]}",
            "prompt": prompt,
            "reference_solution": ""
        })
    
    # Generate responses
    results = []
    for example in formatted_examples:
        print(f"Generating response for example {example['id']}...")
        response = generate_response(model, tokenizer, example["prompt"])
        
        results.append({
            "id": example["id"],
            "prompt": example["prompt"],
            "response": response,
            "reference_solution": example["reference_solution"]
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate REPAIRITY models")
    parser.add_argument("--model_path", required=True, help="Path to the model to evaluate")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples to evaluate")
    parser.add_argument("--output_file", default=None, help="Output file for results")
    parser.add_argument("--claude_test_path", default=None, help="Path to Claude test data")
    
    args = parser.parse_args()
    
    # Try to find Claude test data if not specified
    if args.claude_test_path is None:
        # Look for test data in default location
        model_name = os.path.basename(os.path.normpath(args.model_path))
        if "_explicit" in model_name:
            strategy = "explicit"
        elif "_cot" in model_name:
            strategy = "cot"
        elif "_alternating" in model_name:
            strategy = "alternating"
        else:
            strategy = "none"
            
        default_test_path = f"./data/test/claude_traces_test_set_{strategy}.json"
        if os.path.exists(default_test_path):
            args.claude_test_path = default_test_path
            print(f"Found Claude test data at {default_test_path}")
    
    # Create output filename if not provided
    if args.output_file is None:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        args.output_file = f"./evaluation_results_{model_name}.json"
    
    # Evaluate the model
    results = evaluate_model(args.model_path, args.num_examples, claude_test_path=args.claude_test_path)
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 