import os
import json
import time
import random
import argparse
import anthropic
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import re
import traceback

# Set up argument parser
parser = argparse.ArgumentParser(description="Collect binary labeled data using Qwen2.5-Coder and Claude 3.7")
parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Path to Qwen model (default: Qwen/Qwen2.5-Coder-7B-Instruct)")
parser.add_argument("--dataset_name", type=str, default="bigcode/bigcodebench", help="Dataset name")
parser.add_argument("--dataset_version", type=str, default="v0.1.4", help="Dataset version")
parser.add_argument("--split", type=str, default="test", help="Dataset split")
parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
parser.add_argument("--output_dir", type=str, default="./preference_data", help="Output directory")
parser.add_argument("--output_file", type=str, default=None, help="Output file path")
parser.add_argument("--claude_api_key", type=str, required=True, help="Claude API key")
parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
args = parser.parse_args()

# Handle output file path
if args.output_file is None:
    args.output_file = os.path.join(args.output_dir, "binary_feedback_data.json")

# Create output directory
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

# Set up logging
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(args.output_file), "collect_binary_feedback.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=args.claude_api_key)

# Load the model and tokenizer
logger.info(f"Loading model from {args.model_path}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Load the dataset
logger.info(f"Loading dataset {args.dataset_name}/{args.dataset_version}...")
try:
    try:
        dataset = load_dataset(args.dataset_name, "default")
        logger.info(f"Successfully loaded dataset from Hugging Face")
    except ValueError as e:
        logger.warning(f"Could not load dataset with specified config: {e}")
        logger.info(f"Trying to load dataset without version specification...")
        dataset = load_dataset(args.dataset_name)
        logger.info(f"Successfully loaded dataset without version specification")
    
    # Find appropriate split
    if args.split in dataset:
        eval_data = dataset[args.split]
    else:
        # Try to get any available split
        if hasattr(dataset, "keys"):
            available_splits = list(dataset.keys())
            logger.warning(f"Split {args.split} not found. Using {available_splits[0]} instead.")
            eval_data = dataset[available_splits[0]]
        else:
            logger.warning(f"No splits found in dataset. Using the dataset directly.")
            eval_data = dataset
    
    logger.info(f"Loaded dataset with {len(eval_data)} examples")
    
except Exception as e:
    logger.error(f"Error loading dataset from Hugging Face: {e}")
    logger.info(f"Attempting to load from collected data...")
    
    # Look for collected data
    collected_data_path = os.path.join("data", "claude_reasoning_traces_v0.1.4_100.json")
    if not os.path.exists(collected_data_path):
        logger.error(f"Could not find collected data at {collected_data_path}")
        raise
    
    logger.info(f"Loading data from {collected_data_path}")
    with open(collected_data_path, 'r') as f:
        collected_data = json.load(f)
    
    # Convert to dataset format
    examples = []
    for item in collected_data:
        examples.append({
            "task_id": item.get("task_id", ""),
            "task": item.get("task", ""),
            "solution": item.get("solution", ""),
            "reference_solution": item.get("reference_solution", "")
        })
    
    from datasets import Dataset
    eval_data = Dataset.from_list(examples)
    logger.info(f"Created dataset from collected data with {len(eval_data)} examples")

# Sample data points
if args.num_samples < len(eval_data):
    random.seed(42)
    selected_indices = random.sample(range(len(eval_data)), args.num_samples)
    eval_samples = [eval_data[i] for i in selected_indices]
else:
    eval_samples = [eval_data[i] for i in range(len(eval_data))]

logger.info(f"Selected {len(eval_samples)} samples for evaluation")

# Helper function to extract prompts from various dataset formats
def extract_prompt(example):
    if isinstance(example, dict):
        # Try different field names for prompts
        for field in ["prompt", "instruction", "input", "question", "instruct_prompt", "code_prompt", "complete_prompt", "task"]:
            if field in example and example[field]:
                return example[field]
        
        # If no specific field found, try any string field with sufficient length
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 20:
                return value
    
    # Fallback to string representation
    return str(example)

# Helper function to generate a single completion using Qwen2.5-Coder-7B-Instruct
def generate_completion(prompt, temperature=0.2):
    try:
        # Format as a chat message
        system_prompt = "You are a skilled programming assistant. Your task is to provide correct, efficient, and well-documented code solutions."
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Check if the model supports apply_chat_template
        if hasattr(tokenizer, 'apply_chat_template'):
            chat_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        else:
            # Fallback for models without chat template
            chat_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        # Prepare inputs
        inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract the generated part
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "assistant" in full_text.lower():
            # Find the assistant's response in the chat template
            match = re.search(r'(?:assistant|bot):\s*(.*)', full_text, re.IGNORECASE | re.DOTALL)
            if match:
                generated_text = match.group(1).strip()
            else:
                # Just extract everything after the prompt
                prompt_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
                generated_text = full_text[prompt_len:].strip()
        else:
            # Just extract everything after the prompt
            prompt_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            generated_text = full_text[prompt_len:].strip()
        
        return generated_text.strip()
            
    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error generating completion: {str(e)}"

# Claude evaluation prompt template for binary feedback
def get_claude_evaluation_prompt(prompt, completion):
    eval_prompt = f"""I'll show you a programming task and a solution generated by the Qwen2.5-Coder-7B-Instruct model. 
                Your job is to evaluate the solution and provide a binary label of "GOOD" or "BAD".
                
                # Programming Task:
                {prompt}
                
                # Solution:
                {completion}
                
                # Evaluation Criteria:
                1. Correctness: Does the solution correctly solve the problem?
                2. Efficiency: Is the solution efficient in terms of time and space complexity?
                3. Readability: Is the code well-structured, properly commented, and easy to understand?
                4. Robustness: Does the solution handle edge cases and potential errors?

                # Your Evaluation:
                First, provide a brief analysis of the solution (2-3 sentences).
                Then, provide your final verdict as either "GOOD" or "BAD".
                
                Format your response as:
                
                Analysis: [Your analysis here]
                
                Verdict: [GOOD/BAD]
                """
    
    return eval_prompt

# Function to parse Claude's evaluation and extract binary label
def parse_claude_evaluation(response):
    try:
        # Extract the verdict (GOOD or BAD)
        verdict_match = re.search(r'[Vv]erdict:?\s*(GOOD|BAD)', response, re.IGNORECASE)
        
        if verdict_match:
            verdict = verdict_match.group(1).upper()
        else:
            # Try alternative pattern matching
            if re.search(r'\b(good|excellent|great|correct|appropriate)\b', response, re.IGNORECASE):
                verdict = "GOOD"
            elif re.search(r'\b(bad|poor|incorrect|problematic|issues)\b', response, re.IGNORECASE):
                verdict = "BAD"
            else:
                verdict = None
        
        # Extract analysis
        analysis_match = re.search(r'[Aa]nalysis:?\s*(.*?)(?:\n\n[Vv]erdict|$)', response, re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else ""
        
        return {
            "label": verdict,
            "analysis": analysis,
            "full_evaluation": response
        }
    
    except Exception as e:
        logger.error(f"Error parsing evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "label": None,
            "analysis": "",
            "error": str(e),
            "full_evaluation": response
        }

# Main data collection loop
feedback_data = []

for i, sample in enumerate(tqdm(eval_samples)):
    sample_id = sample.get("task_id", f"sample_{i}")
    
    try:
        # Extract the prompt
        prompt = extract_prompt(sample)
        logger.info(f"Sample {i} ({sample_id}): Extracted prompt of length {len(prompt)}")
        
        # Generate a single completion using Qwen
        completion = generate_completion(prompt, args.temperature)
        logger.info(f"Generated completion from Qwen2.5-Coder-7B-Instruct model ({len(completion)} chars)")
        
        # Skip if error in generation
        if "Error generating completion" in completion:
            logger.warning(f"Skipping sample {i} due to generation error")
            continue
        
        # Prepare the evaluation prompt for Claude
        eval_prompt = get_claude_evaluation_prompt(prompt, completion)
        
        # Get Claude's evaluation
        try:
            logger.info(f"Sending evaluation request to Claude for sample {i}")
            claude_response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": eval_prompt}
                ]
            )
            
            claude_evaluation = claude_response.content[0].text
            logger.info(f"Received evaluation from Claude ({len(claude_evaluation)} chars)")
            
            # Parse the evaluation
            parsed_eval = parse_claude_evaluation(claude_evaluation)
            
            # Skip if no verdict
            if not parsed_eval["label"]:
                logger.warning(f"No clear verdict for sample {i}, skipping")
                continue
            
            # Log the parsed results
            logger.info(f"Parsed label: {parsed_eval['label']}")
            
            # Create feedback data entry
            feedback_entry = {
                "sample_id": sample_id,
                "input": prompt,
                "solution": completion,
                "label": parsed_eval["label"],
                "analysis": parsed_eval["analysis"],
                "full_evaluation": parsed_eval["full_evaluation"]
            }
            feedback_data.append(feedback_entry)
            logger.info(f"Added feedback entry with label: {parsed_eval['label']}")
            
            # Save interim results frequently
            if (i + 1) % 10 == 0 or (i + 1) == len(eval_samples):
                logger.info(f"Saving interim results after {i+1} samples")
                with open(f"{os.path.dirname(args.output_file)}/binary_feedback_interim_{i+1}.json", 'w') as f:
                    json.dump(feedback_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error with Claude API for sample {i}: {str(e)}")
            logger.error(traceback.format_exc())
            time.sleep(5)  # Back off on API errors
    
    except Exception as e:
        logger.error(f"Error processing sample {i}: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Sleep to avoid API rate limits
    time.sleep(2)

# Save final results
logger.info(f"Data collection completed. Collected {len(feedback_data)} binary feedback entries.")
with open(args.output_file, 'w') as f:
    json.dump(feedback_data, f, indent=2)

# Save summary statistics
stats = {
    "total_samples_processed": len(eval_samples),
    "total_feedback_entries": len(feedback_data),
    "good_label_count": sum(1 for entry in feedback_data if entry["label"] == "GOOD"),
    "bad_label_count": sum(1 for entry in feedback_data if entry["label"] == "BAD")
}

with open(f"{os.path.dirname(args.output_file)}/collection_summary.json", 'w') as f:
    json.dump(stats, f, indent=2)

logger.info(f"Summary statistics: {stats}")
logger.info("Binary feedback data collection using Qwen2.5-Coder-7B-Instruct and Claude 3.7 completed successfully")

# Function to be called by wrapper script
def collect_preferences(args_dict):
    # Update global args with provided values
    for key, value in args_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    # The script's main functionality is already executed when imported
    return feedback_data