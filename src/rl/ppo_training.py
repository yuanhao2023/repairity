import os
import argparse
import random
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import json
import time
import wandb
import logging
import traceback
import datetime

# TRL imports (Transformer Reinforcement Learning)
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl.core import LengthSampler
except ImportError:
    raise ImportError(
        "Please install the TRL library: pip install trl"
    )

# Set up argument parser
parser = argparse.ArgumentParser(description="Fine-tune Qwen-32B using RL-LF with Claude 3.7's preferences via PPO")
parser.add_argument("--base_model", type=str, default="Qwen/Qwen-32B", help="Base model to fine-tune (default: Qwen/Qwen-32B)")
parser.add_argument("--reward_model", type=str, required=True, help="Path to the reward model")
parser.add_argument("--dataset_name", type=str, default="bigcode/bigcodebench", help="Dataset name")
parser.add_argument("--dataset_version", type=str, default="v0.1.4", help="Dataset version")
parser.add_argument("--split", type=str, default=None, help="Dataset split")
parser.add_argument("--output_dir", type=str, default="./rl_lf_model", help="Output directory")
parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
parser.add_argument("--min_response_length", type=int, default=200, help="Min response length to generate")
parser.add_argument("--max_response_length", type=int, default=800, help="Max response length to generate")
parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient training")
parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
parser.add_argument("--use_wandb", action="store_true", help="Track experiment with Weights & Biases")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to use for RL-LF")
parser.add_argument("--kl_penalty", type=float, default=0.1, help="KL penalty coefficient")
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, "ppo_finetuning.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log args
logger.info(f"Training arguments: {args}")

# Initialize wandb if enabled
if args.use_wandb:
    try:
        run_name = f"rl-lf-{args.base_model.split('/')[-1]}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project="rl-lf-code-model", name=run_name)
        wandb.config.update(args)
        logger.info(f"Initialized WandB run: {run_name}")
    except Exception as e:
        logger.warning(f"Error initializing WandB: {e}. Continuing without tracking.")
        args.use_wandb = False

# Set seed for reproducibility
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
logger.info(f"Set random seed: {seed}")

# Load dataset
logger.info(f"Loading dataset {args.dataset_name}/{args.dataset_version}...")
try:
    dataset = load_dataset(args.dataset_name, args.dataset_version)
    
    # Find appropriate split
    if args.split and args.split in dataset:
        train_data = dataset[args.split]
    elif hasattr(dataset, "keys") and args.dataset_version in dataset:
        version_data = dataset[args.dataset_version]
        if args.split and args.split in version_data:
            train_data = version_data[args.split]
        else:
            available_splits = list(version_data.keys())
            logger.warning(f"Split {args.split} not found. Available splits: {available_splits}")
            train_data = version_data[available_splits[0]]
    else:
        # Try to get any available split
        if hasattr(dataset, "keys"):
            available_splits = list(dataset.keys())
            logger.warning(f"Split not specified or not found. Using {available_splits[0]} instead.")
            train_data = dataset[available_splits[0]]
        else:
            logger.warning(f"No splits found in dataset. Using the dataset directly.")
            train_data = dataset
    
    logger.info(f"Loaded dataset with {len(train_data)} examples")
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Helper function to extract prompts from various dataset formats
def extract_prompt(example):
    if isinstance(example, dict):
        # Try different field names for prompts
        for field in ["prompt", "instruction", "input", "question", "instruct_prompt", "code_prompt", "complete_prompt"]:
            if field in example and example[field]:
                return example[field]
        
        # If no specific field found, try any string field with sufficient length
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 20:
                return value
    
    # Fallback to string representation
    return str(example)

# Sample prompts to use for training
prompts = []
try:
    for i in range(min(len(train_data), args.max_samples)):
        try:
            prompt = extract_prompt(train_data[i])
            if prompt and len(prompt.strip()) > 20:  # Ensure non-empty, meaningful prompts
                prompts.append(prompt)
        except Exception as e:
            logger.warning(f"Error extracting prompt {i}: {e}")
            continue
    
    logger.info(f"Extracted {len(prompts)} prompts for RL-LF training")
except Exception as e:
    logger.error(f"Error processing prompts: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Save the prompts used for training
with open(f"{args.output_dir}/training_prompts.json", 'w') as f:
    json.dump(prompts[:10], f, indent=2)  # Save just a sample of prompts

# Load base model and tokenizer
logger.info(f"Loading base model and tokenizer from {args.base_model}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    logger.info("Base model loaded successfully")
    
    # Apply LoRA if specified
    if args.use_lora:
        logger.info("Applying LoRA adapters...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            # Target modules - may need adjustment based on model architecture
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "wte"]
        )
        model.pretrained_model = get_peft_model(model.pretrained_model, peft_config)
        trainable_params, all_params = model.pretrained_model.get_nb_trainable_parameters()
        logger.info(f"Trainable params: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}%)")
except Exception as e:
    logger.error(f"Error loading base model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Load reward model
logger.info(f"Loading reward model from {args.reward_model}...")
try:
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        num_labels=1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    logger.info("Reward model loaded successfully")
except Exception as e:
    logger.error(f"Error loading reward model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
reward_model.to(device)
logger.info(f"Using device: {device}")

# Define reward function using the trained reward model
def reward_function(prompt_texts, response_texts):
    batch_size = len(prompt_texts)
    rewards = torch.zeros(batch_size, device=device)
    
    for i in range(batch_size):
        try:
            # Combine prompt and response
            full_text = prompt_texts[i] + "\n" + response_texts[i]
            
            # Tokenize
            inputs = reward_tokenizer(
                full_text,
                max_length=args.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Get reward score
            with torch.no_grad():
                reward_outputs = reward_model(**inputs)
                reward_score = reward_outputs.logits.squeeze()
            
            rewards[i] = reward_score
        except Exception as e:
            logger.warning(f"Error computing reward for sample {i}: {e}")
            rewards[i] = 0.0  # Assign zero reward on error
    
    # Log reward statistics
    if batch_size > 0:
        logger.info(f"Rewards - Mean: {rewards.mean().item():.4f}, Min: {rewards.min().item():.4f}, Max: {rewards.max().item():.4f}")
    
    return rewards

# Configure PPO training
ppo_config = PPOConfig(
    model_name=args.base_model,
    learning_rate=args.learning_rate,
    log_with="wandb" if args.use_wandb else None,
    batch_size=args.per_device_train_batch_size,
    mini_batch_size=1,  # Use 1 for smallest memory footprint
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=True,
    target_kl=0.1,
    kl_penalty=args.kl_penalty,
    seed=seed,
    use_score_scaling=True,
    use_score_norm=True,
    score_clip=None,
    ppo_epochs=4
)

# Initialize PPO trainer
try:
    logger.info("Initializing PPO trainer for Qwen-32B RL-LF...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=prompts,  # We'll use the list of prompts as our dataset
        data_collator=None
    )
    logger.info("PPO trainer initialized successfully")
except Exception as e:
    logger.error(f"Error initializing PPO trainer: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Response length sampler
response_length_sampler = LengthSampler(args.min_response_length, args.max_response_length)

# Training loop
logger.info("Starting RL-LF training of Qwen-32B with Claude 3.7's preferences...")
for epoch in range(args.num_train_epochs):
    logger.info(f"Epoch {epoch+1}/{args.num_train_epochs}")
    
    # Get batch of prompts
    for batch_idx, batch in enumerate(ppo_trainer.dataloader):
        try:
            # Get the prompts (batch is a list of strings)
            prompt_texts = batch
            
            # Tokenize prompts
            prompt_tensors = []
            for prompt in prompt_texts:
                # Format as a chat message for better results
                chat = [
                    {"role": "system", "content": "You are a skilled programmer. Write clean, efficient, and correct code."},
                    {"role": "user", "content": prompt}
                ]
                chat_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
                
                tokens = tokenizer(
                    chat_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length // 2  # Leave room for the response
                ).input_ids.squeeze(0).to(device)
                
                prompt_tensors.append(tokens)
            
            # Generate responses
            response_tensors = []
            for prompt_tensor in prompt_tensors:
                try:
                    gen_len = response_length_sampler()
                    generation = ppo_trainer.generate(
                        prompt_tensor.unsqueeze(0),
                        max_new_tokens=gen_len,
                        do_sample=True,
                        temperature=1.0,
                        top_k=0,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    
                    # Extract only the generated part (remove the prompt)
                    response = generation[0][len(prompt_tensor):]
                    response_tensors.append(response)
                except Exception as e:
                    logger.warning(f"Error during generation: {e}")
                    # Create a dummy response
                    response_tensors.append(torch.tensor([tokenizer.eos_token_id], device=device))
            
            # Decode responses
            response_texts = []
            for response in response_tensors:
                try:
                    text = tokenizer.decode(response, skip_special_tokens=True)
                    response_texts.append(text)
                except Exception as e:
                    logger.warning(f"Error decoding response: {e}")
                    response_texts.append("")
            
            # Calculate rewards using the Claude-trained reward model
            rewards = reward_function(prompt_texts, response_texts)
            
            # Run PPO step
            try:
                stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
                
                # Print stats
                if batch_idx % 5 == 0:
                    stats_to_log = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in stats.items()}
                    logger.info(f"Batch {batch_idx}, Stats: {stats_to_log}")
                    
                    # Save a generated example for inspection
                    example_idx = 0  # First example in batch
                    example_info = {
                        "prompt": prompt_texts[example_idx][:200] + "...",
                        "response": response_texts[example_idx][:200] + "...",
                        "reward": float(rewards[example_idx].item())
                    }
                    logger.info(f"Example - Prompt: {example_info['prompt']}")
                    logger.info(f"Example - Response: {example_info['response']}")
                    logger.info(f"Example - Reward: {example_info['reward']}")
                    
                    # Log to wandb
                    if args.use_wandb:
                        wandb.log(stats_to_log)
                        wandb.log({
                            "example_reward": example_info["reward"],
                            "avg_reward": rewards.mean().item()
                        })
            except Exception as e:
                logger.error(f"Error in PPO step: {str(e)}")
                logger.error(traceback.format_exc())
                continue
            
            # Save checkpoint periodically
            if batch_idx > 0 and batch_idx % 50 == 0:
                checkpoint_dir = f"{args.output_dir}/checkpoint-epoch-{epoch+1}-batch-{batch_idx}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model
                try:
                    if args.use_lora:
                        model.pretrained_model.save_pretrained(checkpoint_dir)
                    else:
                        model.save_pretrained(checkpoint_dir)
                    
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
                    
                    # Save PPO trainer stats
                    with open(f"{checkpoint_dir}/ppo_stats.json", 'w') as f:
                        # Convert any tensors to floats for JSON serialization
                        json_stats = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in stats.items()}
                        json.dump(json_stats, f, indent=2)
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {str(e)}")
                    logger.error(traceback.format_exc())
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            logger.error(traceback.format_exc())
            continue

# Save final model
logger.info(f"Saving final model to {args.output_dir}...")
try:
    if args.use_lora:
        model.pretrained_model.save_pretrained(args.output_dir)
    else:
        model.save_pretrained(args.output_dir)
    
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")
except Exception as e:
    logger.error(f"Error saving final model: {str(e)}")
    logger.error(traceback.format_exc())

# Save summary
training_summary = {
    "completed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "args": vars(args),
    "num_prompts": len(prompts),
    "num_epochs": args.num_train_epochs,
    "device": device,
    "model_path": args.output_dir
}

with open(f"{args.output_dir}/training_summary.json", 'w') as f:
    json.dump(training_summary, f, indent=2)

logger.info("RL-LF training of Qwen-32B with Claude 3.7's preferences completed!")