#!/usr/bin/env python
# Eval.py - Evaluation script for SWE-bench

import os
import json
import time
import argparse
import torch
import numpy as np
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on SWE-bench")
    
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="Name or path of the base model")
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Verified",
                        help="SWE-bench dataset to evaluate on")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for generation sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--load_in_4bit", action="store_true", 
                        help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", 
                        help="Load model in 8-bit quantization")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate on")
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_model_and_tokenizer(args):
    """Load the model and tokenizer."""
    print(f"Loading model from {args.model_path} with base model {args.base_model}...")
    
    # Determine if the model is a full model or a LoRA adapter
    is_lora = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    
    # Set up quantization config if needed
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
    
    # Load the tokenizer from the output directory or base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )
    except:
        print(f"Could not load tokenizer from {args.model_path}, falling back to base model")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True
        )
    
    # Make sure we have the pad token set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the base model first
    if is_lora:
        print("Loading base model for LoRA...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Then load the LoRA adapter
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(
            model,
            args.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    else:
        # Load the full model directly
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
    
    return model, tokenizer

def load_test_data(args):
    """Load the test dataset."""
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset)
    
    if args.split not in dataset:
        print(f"Split {args.split} not found in dataset. Available splits: {list(dataset.keys())}")
        return None
    
    test_data = dataset[args.split]
    
    # Limit number of samples if specified
    if args.num_samples > 0 and args.num_samples < len(test_data):
        np.random.seed(args.seed)
        indices = np.random.choice(len(test_data), args.num_samples, replace=False)
        test_data = test_data.select(indices)
        print(f"Selected {args.num_samples} random samples for evaluation")
    
    return test_data

def format_prompt(example):
    """Format a SWE-bench example for model input."""
    repo = example.get("repo", "")
    problem_statement = example.get("problem_statement", "")
    
    # Create system prompt
    system = "You are a skilled programmer. Fix software bugs and implement features to pass failing tests."
    
    # Create user prompt
    user_prompt = f"Repository: {repo}\n\nProblem to solve:\n{problem_statement}"
    
    return system, user_prompt

def extract_patch_from_response(response):
    """Extract code patch from model response."""
    # Try to extract code between markdown code blocks
    if "```" in response:
        # Look for ```diff blocks first
        diff_pattern = re.compile(r'```diff\s*(.*?)```', re.DOTALL)
        diff_match = diff_pattern.search(response)
        if diff_match:
            return diff_match.group(1).strip()
        
        # Otherwise look for any code blocks
        parts = response.split("```")
        if len(parts) >= 3:  # At least one code block
            # Return the first code block (skip language identifier if present)
            code_block = parts[1]
            if "\n" in code_block and not code_block.strip().startswith("\n"):
                # There might be a language identifier on the first line
                code_block = "\n".join(code_block.split("\n")[1:])
            return code_block.strip()
    
    # If no code block markers found, return the entire response
    return response.strip()

def is_patch_resolved(generated_patch, ground_truth_patch):
    """Determine if the generated patch resolves the issue.
    
    This is a simple heuristic - a more accurate approach would involve
    running the patch against the tests.
    """
    if not generated_patch or not ground_truth_patch:
        return False
    
    # Remove whitespace and common code markers for comparison
    def normalize(patch):
        patch = re.sub(r'\s+', '', patch)
        patch = re.sub(r'[+-]', '', patch)
        return patch.lower()
    
    # Check if there's significant overlap in the normalized patches
    gen_norm = normalize(generated_patch)
    truth_norm = normalize(ground_truth_patch)
    
    # Simple heuristic: check if there are some common elements
    # A more sophisticated approach would be preferred in production
    min_len = min(len(gen_norm), len(truth_norm))
    if min_len > 0:
        # Check for at least some overlapping content
        common_chars = sum(1 for a, b in zip(gen_norm, truth_norm) if a == b)
        similarity = common_chars / min_len
        return similarity > 0.5  # Arbitrary threshold
    
    return False

def evaluate_model(model, tokenizer, test_data, args):
    """Evaluate the model on the test data."""
    results = []
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare the model for evaluation
    model.eval()
    
    # Track metrics
    total_time = 0
    total_examples = 0
    resolved_count = 0
    
    # Create progress bar
    pbar = tqdm(total=len(test_data), desc="Evaluating")
    
    # For each test example
    for idx, example in enumerate(test_data):
        # Format the prompt
        system, user_prompt = format_prompt(example)
        
        # Create chat template
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        
        # Track time
        start_time = time.time()
        
        # Generate response
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        # Generate with specified parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode the response
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (not the prompt)
        response = full_output[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
        
        # Calculate time
        generation_time = time.time() - start_time
        total_time += generation_time
        total_examples += 1
        
        # Extract ground truth patch
        ground_truth = example.get("patch", "")
        
        # Extract patch from response
        generated_patch = extract_patch_from_response(response)
        
        # Check if resolved
        is_resolved = is_patch_resolved(generated_patch, ground_truth)
        if is_resolved:
            resolved_count += 1
        
        # Store results
        result = {
            "example_id": example.get("instance_id", f"example_{idx}"),
            "repo": example.get("repo", ""),
            "commit_sha": example.get("base_commit", "N/A"),
            "prompt": user_prompt,
            "generated_patch": generated_patch,
            "ground_truth_patch": ground_truth,
            "generation_time_seconds": generation_time,
            "difficulty": example.get("difficulty", "unknown"),
            "resolved": is_resolved
        }
        
        results.append(result)
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({"avg_time": total_time / total_examples, "resolved": f"{resolved_count}/{total_examples}"})
        
        # Save intermediate results every 10 examples
        if (idx + 1) % 10 == 0:
            with open(os.path.join(args.output_dir, f"intermediate_results_{idx+1}.json"), "w") as f:
                json.dump(results, f, indent=2)
    
    pbar.close()
    
    # Calculate metrics
    avg_time = total_time / total_examples if total_examples > 0 else 0
    resolved_ratio = resolved_count / total_examples if total_examples > 0 else 0
    
    # Return results and metrics
    return results, {
        "avg_generation_time": avg_time, 
        "total_examples": total_examples,
        "resolved_count": resolved_count,
        "resolved_ratio": resolved_ratio
    }

def write_evaluation_report(results, metrics, args):
    """Write the evaluation report."""
    # Save full results
    with open(os.path.join(args.output_dir, "full_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Write summary report
    total_examples = len(results)
    difficulties = {}
    resolved_by_difficulty = {}
    
    # Count examples by difficulty and track resolved ratio
    for result in results:
        difficulty = result.get("difficulty", "unknown")
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        if difficulty not in resolved_by_difficulty:
            resolved_by_difficulty[difficulty] = {"resolved": 0, "total": 0}
        
        resolved_by_difficulty[difficulty]["total"] += 1
        if result.get("resolved", False):
            resolved_by_difficulty[difficulty]["resolved"] += 1
    
    # Write markdown report
    with open(os.path.join(args.output_dir, "evaluation_report.md"), "w") as f:
        f.write("# SWE-bench Evaluation Report\n\n")
        
        f.write("## Model Information\n")
        f.write(f"- **Base Model:** {args.base_model}\n")
        f.write(f"- **Fine-tuned Model:** {args.model_path}\n")
        f.write(f"- **Dataset:** {args.dataset}\n")
        f.write(f"- **Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Evaluation Metrics\n")
        f.write(f"- **Total Examples:** {total_examples}\n")
        f.write(f"- **Resolved Examples:** {metrics['resolved_count']} ({metrics['resolved_ratio']*100:.2f}%)\n")
        f.write(f"- **Average Generation Time:** {metrics['avg_generation_time']:.2f} seconds\n\n")
        
        f.write("## Examples by Difficulty\n")
        for difficulty, count in sorted(difficulties.items()):
            resolved = resolved_by_difficulty.get(difficulty, {"resolved": 0, "total": 0})
            resolved_ratio = resolved["resolved"] / resolved["total"] if resolved["total"] > 0 else 0
            f.write(f"- **{difficulty}:** {count} examples ({count/total_examples*100:.1f}%) - Resolved: {resolved['resolved']}/{resolved['total']} ({resolved_ratio*100:.1f}%)\n")
        
        f.write("\n## All Examples with Patches\n\n")
        
        # Show all examples with their patches
        for i, result in enumerate(results):
            f.write(f"### Example {i+1}: {result['repo']}\n\n")
            f.write(f"**Instance ID:** {result['example_id']}\n")
            f.write(f"**Commit SHA:** `{result['commit_sha']}`\n")
            f.write(f"**Difficulty:** {result['difficulty']}\n")
            f.write(f"**Resolved:** {'Yes' if result.get('resolved', False) else 'No'}\n\n")
            
            f.write("**Problem:**\n")
            f.write(f"```\n{result['prompt']}\n```\n\n")
            
            f.write("**Generated Patch:**\n")
            f.write(f"```diff\n{result['generated_patch']}\n```\n\n")
            
            f.write("**Ground Truth Patch:**\n")
            f.write(f"```diff\n{result['ground_truth_patch']}\n```\n\n")
            
            f.write(f"**Generation Time:** {result['generation_time_seconds']:.2f} seconds\n\n")
            
            if i < len(results) - 1:
                f.write("---\n\n")
    
    print(f"Evaluation report saved to {os.path.join(args.output_dir, 'evaluation_report.md')}")
    print(f"Resolved ratio: {metrics['resolved_ratio']*100:.2f}% ({metrics['resolved_count']}/{metrics['total_examples']})")

def main():
    args = parse_arguments()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load test data
    test_data = load_test_data(args)
    if test_data is None:
        print("Failed to load test data. Exiting...")
        return
    
    # Evaluate the model
    print(f"Evaluating on {len(test_data)} examples...")
    results, metrics = evaluate_model(model, tokenizer, test_data, args)
    
    # Write evaluation report
    write_evaluation_report(results, metrics, args)
    
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()

# python SwebenchEval.py --model_path ./qwen2.5-coder-7b-swebench-finetuned \
#                --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
#                --load_in_4bit \
#                --output_dir ./evaluation_results \
#                --num_samples 50 \
#                --temperature 0.1