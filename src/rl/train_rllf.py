#!/usr/bin/env python3
# train_rllf.py - Reinforcement Learning from LLM Feedback using PPO

import os
import json
import torch
import random
import argparse
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import (
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with RLLF using PPO")
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
        default="./models/qwen_rllf",
        help="Output directory for RLLF fine-tuned model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--mini_batch_size", 
        type=int, 
        default=1,
        help="Mini batch size for PPO updates"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_prompt_length", 
        type=int, 
        default=512,
        help="Maximum prompt length"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs (PPO epochs)"
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
        help="Load policy model in 8-bit mode"
    )
    parser.add_argument(
        "--load_in_4bit", 
        action="store_true",
        help="Load policy model in 4-bit mode"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100,
        help="Number of samples to use from dataset"
    )
    parser.add_argument(
        "--ppo_steps", 
        type=int, 
        default=10,
        help="Number of PPO update steps"
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
    
    # Format dataset
    formatted_data = []
    for item in data:
        # Assuming each item has 'input' field
        formatted_data.append({
            "query": item.get("input", ""),
            "input": item.get("input", "")
        })
    
    # Sample if requested
    if num_samples and num_samples < len(formatted_data):
        logger.info(f"Sampling {num_samples} examples from dataset")
        formatted_data = random.sample(formatted_data, num_samples)
    
    return formatted_data

def load_reward_model(reward_model_path: str):
    """Load the binary reward model."""
    logger.info(f"Loading reward model from {reward_model_path}")
    
    # Try to load as a PEFT model first
    try:
        config = PeftConfig.from_pretrained(reward_model_path)
        logger.info(f"Loading base model: {config.base_model_name_or_path}")
        
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=2,
            device_map="auto"
        )
        reward_model = PeftModel.from_pretrained(reward_model, reward_model_path)
        logger.info(f"Loaded PEFT reward model based on {config.base_model_name_or_path}")
    except Exception as e:
        logger.info(f"Loading as standard model: {str(e)}")
        # Fall back to regular model loading
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    return reward_model, tokenizer

def get_reward(
    reward_model,
    reward_tokenizer,
    batch_inputs,
    batch_outputs,
    max_length=1024
):
    """Get rewards from the reward model."""
    rewards = []
    
    for input_text, output_text in zip(batch_inputs, batch_outputs):
        # Format as expected by reward model
        combined_text = f"Task: {input_text}\n\nSolution: {output_text}"
        
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
            
            # Get "GOOD" class probability as reward
            reward = probabilities[0][1].item()  # Probability of class 1 (GOOD)
            rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float)

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
    
    # Prepare model for PPO
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(policy_model)
    
    # If using quantization, we need to use PEFT
    if args.load_in_4bit or args.load_in_8bit:
        logger.info("Setting up LoRA for policy model")
        policy_model.pretrained_model = prepare_model_for_kbit_training(policy_model.pretrained_model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        policy_model.pretrained_model = get_peft_model(
            policy_model.pretrained_model, 
            peft_config
        )
        
        policy_model.pretrained_model.print_trainable_parameters()
    
    # Define PPO configuration with minimal parameters
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate
    )
    
    # Define generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        pad_token_id=policy_tokenizer.pad_token_id,
        eos_token_id=policy_tokenizer.eos_token_id,
        do_sample=True
    )
    
    # Initialize the PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        tokenizer=policy_tokenizer,
        dataset=dataset,
    )
    
    # Training loop
    logger.info("Starting RLLF training with PPO")
    for epoch in range(args.ppo_steps):
        logger.info(f"Starting epoch {epoch+1}/{args.ppo_steps}")
        
        # Get a batch from the dataset
        batch = next(ppo_trainer.dataloader)
        query_tensors = [
            policy_tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length
            ).input_ids.to(policy_model.device).squeeze()
            for query in batch["query"]
        ]
        
        # Generate responses
        response_tensors = []
        for query_tensor in query_tensors:
            response = ppo_trainer.generate(
                query_tensor.unsqueeze(0),
                generation_config=gen_config
            )
            response_tensors.append(response.squeeze())
        
        # Decode responses
        batch_inputs = [policy_tokenizer.decode(query_tensor) for query_tensor in query_tensors]
        batch_outputs = []
        
        for response_tensor, query_tensor in zip(response_tensors, query_tensors):
            # Only decode the newly generated tokens
            if len(response_tensor) > len(query_tensor):
                new_tokens = response_tensor[len(query_tensor):]
                output = policy_tokenizer.decode(new_tokens, skip_special_tokens=True)
                batch_outputs.append(output)
            else:
                # If no new tokens were generated
                batch_outputs.append("")
        
        # Calculate rewards using the reward model
        rewards = get_reward(
            reward_model,
            reward_tokenizer,
            batch["input"],  # Original task inputs
            batch_outputs,   # Generated solutions
            max_length=args.max_length
        )
        
        # Debug rewards
        logger.info(f"Epoch {epoch+1} rewards: {rewards.tolist()}")
        
        # Update the model with PPO
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        logger.info(f"Epoch {epoch+1} stats: {stats}")
        
        # After each epoch, save a checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save policy model
        ppo_trainer.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    final_model_dir = os.path.join(args.output_dir, "final-model")
    os.makedirs(final_model_dir, exist_ok=True)
    ppo_trainer.save_pretrained(final_model_dir)
    logger.info(f"Saved final model to {final_model_dir}")
    
    # Also save model without ValueHead for easier inference
    unwrapped_model = ppo_trainer.model.pretrained_model
    unwrapped_model.save_pretrained(os.path.join(args.output_dir, "unwrapped-model"))
    policy_tokenizer.save_pretrained(os.path.join(args.output_dir, "unwrapped-model"))
    logger.info(f"Saved unwrapped model to {os.path.join(args.output_dir, 'unwrapped-model')}")
    
    logger.info("RLLF training complete!")

if __name__ == "__main__":
    main() 