#!/usr/bin/env python3
# evaluate_bigcodebench.py - Evaluate models on BigCodeBench

import os
import json
import torch
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on BigCodeBench")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model to evaluate"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name to use for this model in results (e.g. 'pure_qwen', 'sft_qwen', 'rllf_qwen')"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v0.1.4",
        help="Version of BigCodeBench to evaluate on"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results/bigcodebench",
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
    parser.add_argument(
        "--num_examples",
        type=int,
        default=-1,
        help="Number of examples to evaluate. -1 means all examples."
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Randomly sample examples instead of taking first N"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling examples"
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

def load_bigcodebench_dataset(version: str, num_examples: int = -1, random_sample: bool = False, seed: int = 42) -> List[Dict]:
    """Load BigCodeBench dataset from Hugging Face."""
    logger.info(f"Loading BigCodeBench dataset version {version}")
    
    # Load dataset
    dataset = load_dataset("bigcode/bigcodebench")
    
    if version not in dataset:
        raise ValueError(f"Version {version} not found in dataset. Available versions: {list(dataset.keys())}")
    
    # Get the dataset for the specified version
    version_data = dataset[version]
    logger.info(f"Total examples in {version}: {len(version_data)}")
    
    # Sample examples if needed
    if num_examples > 0 and num_examples < len(version_data):
        if random_sample:
            # Set seed for reproducibility
            torch.manual_seed(seed)
            indices = torch.randperm(len(version_data))[:num_examples].tolist()
        else:
            indices = list(range(num_examples))
        examples = version_data.select(indices)
    else:
        examples = version_data
    
    return examples

def format_prompt(problem: Dict[str, Any]) -> str:
    """Format a BigCodeBench problem into a prompt."""
    prompt = f"""Task: Please solve the following programming problem. Provide your solution with a clear explanation of your approach.

Problem: {problem['instruct_prompt']}

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
            "problem_id": problem["task_id"],
            "prompt": prompt,
            "solution": solution,
            "canonical_solution": problem["canonical_solution"],
            "has_reasoning": any(phrase in solution.lower() for phrase in [
                "let's approach this step by step",
                "here's my reasoning",
                "let's think about this",
                "first, we need to",
                "the approach is to",
                "let me explain",
                "here's how we'll solve this",
                "the strategy is"
            ])
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
        "avg_solution_length": sum(len(r["solution"]) for r in results) / len(results),
        "reasoning_patterns": {
            "step_by_step": sum(1 for r in results if "step by step" in r["solution"].lower()),
            "here_is_my_reasoning": sum(1 for r in results if "here's my reasoning" in r["solution"].lower()),
            "lets_think": sum(1 for r in results if "let's think" in r["solution"].lower()),
            "first_we": sum(1 for r in results if "first, we" in r["solution"].lower()),
            "approach": sum(1 for r in results if "the approach is" in r["solution"].lower()),
            "strategy": sum(1 for r in results if "the strategy is" in r["solution"].lower())
        }
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
        if isinstance(value, dict):
            logger.info(f"  {metric}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {metric}: {value}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load BigCodeBench dataset
    problems = load_bigcodebench_dataset(
        args.version,
        args.num_examples,
        args.random_sample,
        args.seed
    )
    
    # Evaluate model
    logger.info(f"Evaluating model: {args.model_name}")
    model, tokenizer = load_model(args.model_path, args.load_in_4bit)
    results = evaluate_model(model, tokenizer, problems, args.max_new_tokens)
    save_results(results, args.output_dir, args.model_name)

if __name__ == "__main__":
    main() 