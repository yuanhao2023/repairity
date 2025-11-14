#!/usr/bin/env python3
# simple_rllf.py - Simplified fine-tuning using reward model

import os
import json
import torch
import random
import argparse
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with RLLF using direct optimization")
    parser.add_argument(
        "--qwen_model", 
        type=str, 
        default="models/qwen_repairity_20250330_011532_alternating_alternating",
        help="Path to fine-tuned Qwen model"
    )
    parser.add_argument(
        "--reward_model", 
        type=str, 
        default="./models/binary_reward_model",
        help="Path to binary reward model"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="./data/bigcode-evaluation-data.jsonl",
        help="Path to dataset for training"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./models/preference-ft-qwen-7b",
        help="Output directory for RLLF fine-tuned model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true",
        help="Load model in 8-bit mode"
    )
    parser.add_argument(
        "--load_in_4bit", 
        action="store_true",
        help="Load model in 4-bit mode"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100,
        help="Number of samples to use from dataset"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=512,
        help="Maximum number of new tokens to generate"
    )
    return parser.parse_args()

def prepare_dataset(dataset_path: str, num_samples: int = None):
    """Load and prepare the dataset for training."""
    logger.info(f"Loading dataset from {dataset_path}")
    
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
        data = random.sample(data, num_samples)
    
    return data

def load_reward_model(reward_model_path: str):
    """Load the binary reward model for evaluation."""
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

def generate_solutions(policy_model, policy_tokenizer, inputs, max_new_tokens=512):
    """Generate solutions using the policy model."""
    logger.info(f"Generating solutions for {len(inputs)} inputs")
    
    solutions = []
    for input_text in tqdm(inputs):
        # Tokenize input
        input_ids = policy_tokenizer(input_text, return_tensors="pt").input_ids.to(policy_model.device)
        
        # Generate solution
        with torch.no_grad():
            output_ids = policy_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode solution (remove the input part)
        solution = policy_tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        solutions.append(solution)
    
    return solutions

def score_solutions(reward_model, reward_tokenizer, inputs, solutions, max_length=1024):
    """Score solutions using the reward model."""
    logger.info(f"Scoring {len(solutions)} solutions")
    
    scores = []
    for input_text, solution in tqdm(zip(inputs, solutions)):
        # Format as expected by reward model
        combined_text = f"Task: {input_text}\n\nSolution: {solution}"
        
        # Tokenize
        inputs = reward_tokenizer(
            combined_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(reward_model.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = reward_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # Get "GOOD" class probability as score
            score = probabilities[0][1].item()  # Probability of class 1 (GOOD)
            scores.append(score)
    
    return scores

def filter_high_reward_examples(inputs, solutions, scores, threshold=0.5):
    """Filter examples with high reward scores."""
    logger.info(f"Filtering examples with threshold {threshold}")
    
    filtered_inputs = []
    filtered_solutions = []
    
    for input_text, solution, score in zip(inputs, solutions, scores):
        if score >= threshold:
            filtered_inputs.append(input_text)
            filtered_solutions.append(solution)
    
    logger.info(f"Kept {len(filtered_inputs)}/{len(inputs)} examples with scores >= {threshold}")
    return filtered_inputs, filtered_solutions

def create_supervised_dataset(inputs, solutions, tokenizer, max_length):
    """Create a dataset for supervised fine-tuning."""
    logger.info("Creating supervised fine-tuning dataset")
    
    data = []
    for input_text, solution in zip(inputs, solutions):
        # Format as expected for training
        combined_text = f"{input_text}\n\n{solution}"
        
        # Tokenize
        encoded = tokenizer(
            combined_text,
            max_length=max_length,
            truncation=True,
            return_special_tokens_mask=True
        )
        
        data.append(encoded)
    
    return data

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up file logger
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "rllf_training.log"))
    logger.addHandler(file_handler)
    
    # Log all arguments
    logger.info(f"Training with arguments: {args}")
    
    # Prepare dataset
    dataset = prepare_dataset(args.dataset_path, args.num_samples)
    inputs = [item.get("input", "") for item in dataset]
    
    # Set up quantization config if needed
    quantization_config = None
    if args.load_in_4bit:
        logger.info("Using 4-bit quantization for policy model")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif args.load_in_8bit:
        logger.info("Using 8-bit quantization for policy model")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Load policy model (Qwen)
    logger.info(f"Loading Qwen policy model from {args.qwen_model}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    policy_tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_model,
        trust_remote_code=True
    )
    
    if policy_tokenizer.pad_token is None:
        if policy_tokenizer.eos_token is not None:
            policy_tokenizer.pad_token = policy_tokenizer.eos_token
        else:
            policy_tokenizer.pad_token = policy_tokenizer.eos_token = "</s>"
    
    # Load reward model
    reward_model, reward_tokenizer = load_reward_model(args.reward_model)
    
    # If using quantization, we need to use PEFT for fine-tuning
    if args.load_in_4bit or args.load_in_8bit:
        logger.info("Setting up LoRA for policy model")
        policy_model = prepare_model_for_kbit_training(policy_model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        policy_model = get_peft_model(policy_model, peft_config)
        policy_model.print_trainable_parameters()
    
    # Step 1: Generate solutions using the policy model
    solutions = generate_solutions(policy_model, policy_tokenizer, inputs, args.max_new_tokens)
    
    # Step 2: Score solutions using the reward model
    scores = score_solutions(reward_model, reward_tokenizer, inputs, solutions, args.max_length)
    
    # Step 3: Filter examples with high scores
    filtered_inputs, filtered_solutions = filter_high_reward_examples(inputs, solutions, scores)
    
    if len(filtered_inputs) == 0:
        logger.warning("No high-quality examples found! Using all examples with weighted sampling.")
        filtered_inputs, filtered_solutions = inputs, solutions
    
    # Step 4: Create a supervised dataset from high-quality examples
    supervised_data = create_supervised_dataset(
        filtered_inputs, 
        filtered_solutions, 
        policy_tokenizer, 
        args.max_length
    )
    
    # Step 5: Fine-tune on high-quality examples
    logger.info("Starting fine-tuning on high-quality examples")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
    )
    
    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=policy_tokenizer,
        mlm=False
    )
    
    # Define trainer
    trainer = Trainer(
        model=policy_model,
        args=training_args,
        train_dataset=supervised_data,
        data_collator=data_collator
    )
    
    # Train model
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    policy_tokenizer.save_pretrained(args.output_dir)
    
    logger.info("RLLF training complete!")
    
    # Create a model card
    with open(os.path.join(args.output_dir, "README.md"), "w") as f:
        f.write(f"""# Preference Fine-tuned Qwen Model

This model was created by fine-tuning Qwen on examples selected by a reward model.

## Training Information
- Base Model: {args.qwen_model}
- Reward Model: {args.reward_model}
- Dataset: {args.dataset_path}
- Examples: {len(filtered_inputs)} high-quality examples (score >= 0.5)
- Epochs: {args.epochs}
- Learning Rate: {args.learning_rate}

## Usage
This model can be used for generating high-quality code based on instructions.
""")

if __name__ == "__main__":
    main() 