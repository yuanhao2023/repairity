#!/usr/bin/env python3
# evaluate_models.py - Compare multiple models on BigCodeBench

import os
import json
import time
import torch
import argparse
import logging
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from anthropic import Anthropic

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare models on BigCodeBench")
    parser.add_argument(
        "--bigcodebench_path", 
        type=str, 
        default="./data/bigcodebench.jsonl",
        help="Path to BigCodeBench dataset"
    )
    parser.add_argument(
        "--original_qwen_path", 
        type=str, 
        default="./models/qwen_repairity_20250330_011532_alternating_alternating",
        help="Path to original fine-tuned Qwen model"
    )
    parser.add_argument(
        "--rllf_qwen_path", 
        type=str, 
        default="./models/qwen_rllf/unwrapped-model",
        help="Path to RLLF fine-tuned Qwen model"
    )
    parser.add_argument(
        "--use_claude", 
        action="store_true",
        help="Whether to evaluate Claude 3.7 (requires API key)"
    )
    parser.add_argument(
        "--anthropic_api_key", 
        type=str, 
        default=None,
        help="Anthropic API key for Claude 3.7"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for model inference"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=None,
        help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--reward_model_path", 
        type=str, 
        default="./models/binary_reward_model",
        help="Path to reward model for scoring solutions"
    )
    return parser.parse_args()

def load_dataset(dataset_path: str, num_samples: int = None):
    """Load evaluation dataset."""
    logger.info(f"Loading evaluation dataset from {dataset_path}")
    
    if dataset_path.endswith('.jsonl'):
        # Load from JSONL file
        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]
    else:
        # Load from JSON file
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples from dataset")
    
    # Sample if requested
    if num_samples and num_samples < len(data):
        logger.info(f"Sampling {num_samples} examples from dataset")
        import random
        random.seed(42)  # For reproducibility
        data = random.sample(data, num_samples)
    
    return data

def load_reward_model(reward_model_path: str):
    """Load the binary reward model for evaluation."""
    from transformers import AutoModelForSequenceClassification
    from peft import PeftModel, PeftConfig
    
    logger.info(f"Loading reward model from {reward_model_path}")
    
    # Try to load as a PEFT model first
    try:
        config = PeftConfig.from_pretrained(reward_model_path)
        logger.info(f"Loading base model: {config.base_model_name_or_path}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=2,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, reward_model_path)
        logger.info(f"Loaded PEFT reward model based on {config.base_model_name_or_path}")
    except Exception as e:
        logger.info(f"Loading as standard model: {str(e)}")
        # Fall back to regular model loading
        model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    return model, tokenizer

def get_qwen_prediction(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 512
) -> str:
    """Generate a prediction from Qwen model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.95,
        top_k=0,
        do_sample=True
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    # Only return the generated part (skip the prompt)
    prompt_length = inputs.input_ids.shape[1]
    generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    return generated_text

def get_claude_prediction(client, prompt: str) -> str:
    """Generate a prediction from Claude model."""
    MAX_RETRIES = 5
    RETRY_DELAY = 5  # seconds
    
    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1024,
                temperature=0.2,
                system="You are a helpful coding assistant. Generate the requested code following the instructions exactly.",
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Error with Claude API (attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Failed to get prediction from Claude after {MAX_RETRIES} attempts: {e}")
                return f"ERROR: Failed to get Claude prediction after {MAX_RETRIES} attempts."

def evaluate_solution_with_reward_model(
    reward_model, 
    reward_tokenizer, 
    task: str, 
    solution: str
) -> float:
    """Evaluate a solution using the reward model."""
    # Format as expected by reward model
    combined_text = f"Task: {task}\n\nSolution: {solution}"
    
    # Tokenize
    inputs = reward_tokenizer(
        combined_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(reward_model.device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = reward_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        
        # Get "GOOD" class probability as score
        score = probabilities[0][1].item()  # Probability of class 1 (GOOD)
    
    return score

def evaluate_models(args):
    """Evaluate and compare multiple models on BigCodeBench."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up file logger
    file_handler = logging.FileHandler(os.path.join(args.output_dir, f"evaluation_{timestamp}.log"))
    logger.addHandler(file_handler)
    
    # Log all arguments
    logger.info(f"Evaluation with arguments: {args}")
    
    # Load dataset
    dataset = load_dataset(args.bigcodebench_path, args.num_samples)
    
    # Load reward model for scoring
    reward_model, reward_tokenizer = load_reward_model(args.reward_model_path)
    
    # Initialize models
    models_to_evaluate = {}
    
    # Load Original Qwen model
    logger.info(f"Loading original Qwen model from {args.original_qwen_path}")
    original_qwen = AutoModelForCausalLM.from_pretrained(
        args.original_qwen_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    original_qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.original_qwen_path,
        trust_remote_code=True
    )
    models_to_evaluate["Original Qwen"] = (original_qwen, original_qwen_tokenizer)
    
    # Load RLLF Qwen model
    logger.info(f"Loading RLLF Qwen model from {args.rllf_qwen_path}")
    rllf_qwen = AutoModelForCausalLM.from_pretrained(
        args.rllf_qwen_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    rllf_qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.rllf_qwen_path,
        trust_remote_code=True
    )
    models_to_evaluate["RLLF Qwen"] = (rllf_qwen, rllf_qwen_tokenizer)
    
    # Initialize Claude client if requested
    claude_client = None
    if args.use_claude:
        if not args.anthropic_api_key:
            logger.warning("No Anthropic API key provided. Skipping Claude evaluation.")
        else:
            logger.info("Initializing Claude 3.7 client")
            try:
                claude_client = Anthropic(api_key=args.anthropic_api_key)
                models_to_evaluate["Claude 3.7"] = claude_client
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
    
    # Prepare results storage
    results = []
    
    # Evaluate each example
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating examples")):
        task = example.get("input", "")
        example_id = example.get("id", str(idx))
        
        logger.info(f"Evaluating example {example_id}")
        
        # For each model, generate a solution and evaluate it
        for model_name, model_info in models_to_evaluate.items():
            logger.info(f"Getting prediction from {model_name}")
            
            try:
                # Generate solution
                if model_name == "Claude 3.7":
                    solution = get_claude_prediction(model_info, task)
                else:
                    model, tokenizer = model_info
                    solution = get_qwen_prediction(
                        model, 
                        tokenizer, 
                        task, 
                        max_new_tokens=args.max_new_tokens
                    )
                
                # Evaluate solution
                score = evaluate_solution_with_reward_model(
                    reward_model, 
                    reward_tokenizer, 
                    task, 
                    solution
                )
                
                # Save result
                result = {
                    "example_id": example_id,
                    "model": model_name,
                    "task": task,
                    "solution": solution,
                    "score": score
                }
                
                results.append(result)
                logger.info(f"Model: {model_name}, Score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on example {example_id}: {e}")
                # Still record the error
                results.append({
                    "example_id": example_id,
                    "model": model_name,
                    "task": task,
                    "solution": f"ERROR: {str(e)}",
                    "score": 0.0
                })
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, f"detailed_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved detailed results to {results_file}")
    
    # Aggregate and analyze results
    df = pd.DataFrame(results)
    
    # Calculate average scores by model
    avg_scores = df.groupby('model')['score'].agg(['mean', 'std', 'count'])
    logger.info(f"Average scores:\n{avg_scores}")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, f"summary_{timestamp}.csv")
    avg_scores.to_csv(summary_file)
    
    logger.info(f"Saved summary to {summary_file}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='model', y='score')
    plt.title('Average Scores by Model')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(args.output_dir, f"comparison_plot_{timestamp}.png")
    plt.savefig(plot_file)
    
    logger.info(f"Saved comparison plot to {plot_file}")
    
    # Return summary as a nice string
    summary_str = "Model Comparison Summary:\n\n"
    for model, stats in avg_scores.iterrows():
        summary_str += f"{model}:\n"
        summary_str += f"  Mean Score: {stats['mean']:.4f}\n"
        summary_str += f"  Std Dev: {stats['std']:.4f}\n"
        summary_str += f"  Count: {stats['count']}\n\n"
    
    return summary_str

def main():
    args = parse_args()
    summary = evaluate_models(args)
    print("\n" + "="*50 + "\n")
    print(summary)
    print("="*50)

if __name__ == "__main__":
    main() 