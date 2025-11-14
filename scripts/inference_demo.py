#!/usr/bin/env python3
# inference_demo.py - Demonstration of inference on BigCodeBench

import os
import sys
import json
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def format_prompt(example):
    """Format BigCodeBench example for model input."""
    
    # Extract relevant fields
    task_id = example.get("task_id", "")
    code_prompt = example.get("code_prompt", "")
    instruct_prompt = example.get("instruct_prompt", "")
    complete_prompt = example.get("complete_prompt", "")
    
    # Create system prompt
    system = "You are a skilled programmer. Provide complete, efficient, and correct code solutions."
    
    # Choose the best prompt available
    if instruct_prompt:
        instruction = instruct_prompt
    elif complete_prompt:
        instruction = complete_prompt
    elif code_prompt:
        instruction = f"Complete the following code:\n\n{code_prompt}"
    else:
        instruction = f"Write code to solve the following task (ID: {task_id})."
    
    # Format for Qwen model
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    
    return {
        "prompt": prompt,
        "task_id": task_id,
        "instruction": instruction
    }

def run_inference(model_path, num_samples=5, max_length=1024, temperature=0.2, dataset_version="v0.1.4"):
    """Run inference on BigCodeBench examples using the fine-tuned model."""
    
    # Convert relative path to absolute path if needed
    if model_path.startswith('./') or model_path.startswith('../'):
        model_path = os.path.abspath(model_path)
    
    print(f"Loading model from {model_path}...")
    
    # Check if the path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist")
        return
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load in 4-bit to save GPU memory
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("If the model path is a local directory, ensure it contains all required model files.")
        return
    
    print("Loading BigCodeBench dataset...")
    dataset = load_dataset("bigcode/bigcodebench")
    
    if dataset_version not in dataset:
        print(f"Version {dataset_version} not found in dataset. Available versions: {list(dataset.keys())}")
        return
    
    # Get the dataset for the specified version
    version_data = dataset[dataset_version]
    
    # Take a subset of examples for demo
    if num_samples > 0 and num_samples < len(version_data):
        indices = list(range(num_samples))
        examples = version_data.select(indices)
    else:
        examples = version_data
        num_samples = len(examples)
    
    print(f"Running inference on {num_samples} examples...")
    
    results = []
    
    for i, example in enumerate(examples):
        print(f"\nProcessing example {i+1}/{num_samples} (Task ID: {example.get('task_id', 'unknown')})")
        
        # Format the example for model input
        formatted = format_prompt(example)
        prompt = formatted["prompt"]
        
        # Print the prompt
        print("\nInput prompt:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract the assistant's response
        assistant_response = generated_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        
        # Print the model's output
        print("\nModel output:")
        print("-" * 50)
        print(assistant_response)
        print("-" * 50)
        
        # Save the result
        results.append({
            "task_id": formatted["task_id"],
            "instruction": formatted["instruction"],
            "model_output": assistant_response,
            "reference_solution": example.get("canonical_solution", "")
        })
    
    # Save results to file
    os.makedirs("inference_results", exist_ok=True)
    output_file = os.path.join("inference_results", "bigcodebench_inference.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on BigCodeBench examples")
    parser.add_argument("--model_path", required=True, help="Path to the fine-tuned model")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of examples to run inference on")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of generated sequences")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (0 for greedy)")
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model_path,
        num_samples=args.num_samples,
        max_length=args.max_length,
        temperature=args.temperature
    ) 