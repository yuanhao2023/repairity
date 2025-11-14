#!/usr/bin/env python3
# compare_models.py - Compare different versions of Qwen models on BigCodeBench

import os
import json
import torch
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compare different Qwen models on BigCodeBench")
    parser.add_argument(
        "--initial_model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Path to initial Qwen model"
    )
    parser.add_argument(
        "--sft_model",
        type=str,
        default="models/qwen_repairity_20250330_011532_alternating_alternating",
        help="Path to SFT Qwen model (trained with reasoning)"
    )
    parser.add_argument(
        "--rllf_model",
        type=str,
        default="models/preference-ft-qwen-7b",
        help="Path to RLLF Qwen model (trained with preferences)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load models in 4-bit mode"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    return parser.parse_args()

def load_model(model_path: str, load_in_4bit: bool = False):
    """Load a model and its tokenizer."""
    logger.info(f"Loading model from {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_4bit=load_in_4bit
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def format_prompt(problem: Dict[str, Any]) -> str:
    """Format a problem into a prompt."""
    prompt = f"""Task: Please solve the following coding problem. Provide your solution with a clear explanation of your approach.

Problem: {problem['description']}

Input Format: {problem.get('input_format', 'Not specified')}
Output Format: {problem.get('output_format', 'Not specified')}

Examples:
{problem.get('examples', 'No examples provided')}

Please provide your solution with explanation:"""
    
    return prompt

def evaluate_model(model, tokenizer, problems: List[Dict], max_new_tokens: int = 512):
    """Evaluate a model on the given problems."""
    results = []
    
    for problem in tqdm(problems, desc="Evaluating problems"):
        prompt = format_prompt(problem)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate solution
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
        
        # Decode solution
        solution = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Store result
        result = {
            "problem_id": problem.get("problem_id", "unknown"),
            "prompt": prompt,
            "solution": solution,
            "expected_output": problem.get("test_cases", []),
            "has_reasoning": "let's approach this step by step" in solution.lower() or "here's my reasoning" in solution.lower()
        }
        results.append(result)
    
    return results

def save_results(results: List[Dict], output_path: str, model_name: str):
    """Save evaluation results to a file."""
    output_file = os.path.join(output_path, f"{model_name}_results.json")
    
    # Create metrics
    metrics = {
        "total_problems": len(results),
        "problems_with_reasoning": sum(1 for r in results if r["has_reasoning"]),
        "avg_solution_length": sum(len(r["solution"]) for r in results) / len(results)
    }
    
    # Save detailed results and metrics
    with open(output_file, 'w') as f:
        json.dump({
            "metrics": metrics,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Metrics for {model_name}:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load BigCodeBench dataset
    logger.info("Loading HumanEval dataset")
    dataset = load_dataset("openai_humaneval")  # Using HumanEval benchmark
    problems = [
        {
            "problem_id": sample["task_id"],
            "description": sample["prompt"],
            "test_cases": sample["test"],
            "entry_point": sample["entry_point"]
        }
        for sample in dataset["test"]
    ]
    
    # Evaluate initial model
    logger.info("Evaluating initial Qwen model")
    initial_model, initial_tokenizer = load_model(args.initial_model, args.load_in_4bit)
    initial_results = evaluate_model(initial_model, initial_tokenizer, problems, args.max_new_tokens)
    save_results(initial_results, args.output_dir, "initial_qwen")
    del initial_model, initial_tokenizer
    torch.cuda.empty_cache()
    
    # Evaluate SFT model
    logger.info("Evaluating SFT Qwen model")
    sft_model, sft_tokenizer = load_model(args.sft_model, args.load_in_4bit)
    sft_results = evaluate_model(sft_model, sft_tokenizer, problems, args.max_new_tokens)
    save_results(sft_results, args.output_dir, "sft_qwen")
    del sft_model, sft_tokenizer
    torch.cuda.empty_cache()
    
    # Evaluate RLLF model
    logger.info("Evaluating RLLF Qwen model")
    rllf_model, rllf_tokenizer = load_model(args.rllf_model, args.load_in_4bit)
    rllf_results = evaluate_model(rllf_model, rllf_tokenizer, problems, args.max_new_tokens)
    save_results(rllf_results, args.output_dir, "rllf_qwen")
    
    # Save successful examples with reasoning
    successful_examples = []
    for result in rllf_results:
        if result["has_reasoning"]:
            successful_examples.append({
                "problem_id": result["problem_id"],
                "prompt": result["prompt"],
                "solution_with_reasoning": result["solution"]
            })
    
    # Save successful examples
    successful_file = os.path.join(args.output_dir, "successful_reasoning_examples.json")
    with open(successful_file, 'w') as f:
        json.dump(successful_examples, f, indent=2)
    
    logger.info(f"Saved {len(successful_examples)} successful reasoning examples to {successful_file}")

if __name__ == "__main__":
    main() 