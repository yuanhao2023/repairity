import os
import random
import json
import torch
import numpy as np
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from sklearn.model_selection import train_test_split

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Using 7B model directly
OUTPUT_DIR = "./qwen2.5-coder-7b-swebench-finetuned"
DATASET_NAME = "princeton-nlp/SWE-bench_Verified"  # Changed to SWE-bench_Verified
TRAIN_RATIO = 0.9  # 90% of data for training
TEST_USE_RATIO = 0.5  # Using 50% of test data for training
NUM_SAMPLES = 0  # Set to 0 to use all samples
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-5
MAX_LENGTH = 2048
BATCH_SIZE = 4  # Batch size per GPU
GRADIENT_ACCUMULATION_STEPS = 4  # Adjusted for multi-GPU setup
NUM_EPOCHS = 3
FP16 = True
SEED = 42  # Fixed seed for reproducibility

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Set PyTorch to allow memory expansion
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize distributed training
def setup_distributed():
    # Initialize the process group
    try:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.cuda.set_device(local_rank)
            is_main_process = local_rank == 0
            print(f"Initialized distributed training with rank {local_rank}/{world_size}")
        else:
            print("No distributed environment variables found, running in single process mode")
            local_rank = 0
            is_main_process = True
    except Exception as e:
        print(f"Warning: distributed initialization failed with error: {e}, falling back to non-distributed mode")
        local_rank = 0
        is_main_process = True
    
    return local_rank, is_main_process

# Function to format SWE-bench items for instruction fine-tuning
def format_instruction(example):
    """Format SWE-bench examples for instruction tuning."""
    
    # Extract relevant fields from the SWE-bench structure
    repo = example.get("repo", "")
    problem_statement = example.get("problem_statement", "")
    patch = example.get("patch", "")
    
    # Create system prompt
    system = "You are a skilled programmer. Fix software bugs and implement features to pass failing tests."
    
    # Create formatted input - combine repo information and problem statement
    formatted_input = f"Repository: {repo}\n\nProblem to solve:\n{problem_statement}"
    
    # The patch is the ground truth solution
    formatted_output = f"```\n{patch}\n```"
    
    return {
        "system": system,
        "input": formatted_input,
        "output": formatted_output
    }

# Load and prepare the dataset
def prepare_dataset(is_main_process):
    if is_main_process:
        print(f"Loading SWE-bench Verified dataset...")
    
    # Load the dataset
    dataset = load_dataset(DATASET_NAME)
    
    # For SWE-bench_Verified, we want the 'test' split which is the only one available
    if 'test' not in dataset:
        if is_main_process:
            print(f"Test split not found in dataset. Available splits: {list(dataset.keys())}")
        return []
    
    # Get the dataset for the specified split
    data = dataset['test']
    
    if is_main_process:
        print(f"Dataset size: {len(data)}")
    
    # Set random seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Split the data into train and test sets (90% train, 10% test)
    train_indices, test_indices = train_test_split(
        range(len(data)), 
        train_size=TRAIN_RATIO, 
        random_state=SEED
    )
    
    # Select 50% of test data to also include in training
    test_for_train_indices = random.sample(test_indices, int(len(test_indices) * TEST_USE_RATIO))
    
    # Combine original train indices with selected test indices
    all_train_indices = sorted(list(set(train_indices + test_for_train_indices)))
    
    # Create the final training dataset
    train_data = data.select(all_train_indices)
    
    if is_main_process:
        print(f"Original train split size: {len(train_indices)}")
        print(f"Test data added to training: {len(test_for_train_indices)}")
        print(f"Final training data size: {len(train_data)}")
    
    # Limit number of samples if specified
    if NUM_SAMPLES > 0 and NUM_SAMPLES < len(train_data):
        indices = random.sample(range(len(train_data)), NUM_SAMPLES)
        train_data = train_data.select(indices)
        if is_main_process:
            print(f"Reduced to {NUM_SAMPLES} samples for training")
    
    # Format the examples for instruction tuning
    formatted_data = []
    for example in train_data:
        try:
            formatted_example = format_instruction(example)
            formatted_data.append(formatted_example)
        except Exception as e:
            if is_main_process:
                print(f"Error formatting example: {e}")
            continue
    
    # Save indices for reproducibility only in main process
    if is_main_process:
        os.makedirs("./data", exist_ok=True)
        with open("./data/train_test_indices.json", "w") as f:
            json.dump({
                "train_indices": train_indices,
                "test_indices": test_indices,
                "test_for_train_indices": test_for_train_indices,
                "all_train_indices": all_train_indices
            }, f, indent=2)
        
        # Save the formatted dataset to disk
        with open("./data/formatted_data.json", "w") as f:
            json.dump(formatted_data, f, indent=2)
        
        print(f"Successfully formatted {len(formatted_data)} examples for training")
    
    return formatted_data

# Function to prepare the Qwen model with LoRA
def prepare_model(local_rank):
    if local_rank == 0:
        print("Loading Qwen 7B model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Clear GPU cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Try loading with 4-bit quantization first
        if local_rank == 0:
            print("Attempting to load with 4-bit quantization...")
            
        # Create 4-bit quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # For distributed training, we need to handle device mapping differently
        # In distributed mode, put model on the local rank's GPU
        # Avoid using 'auto' device_map in distributed settings
        if dist.is_initialized():
            device_map = {"": local_rank}
        else:
            device_map = "auto"
        
        if local_rank == 0:
            print(f"Using device_map: {device_map}")
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map
        )
        
    except Exception as e:
        if local_rank == 0:
            print(f"Failed to load with 4-bit quantization: {e}")
            print("Falling back to FP16 without quantization...")
        
        # Load model without quantization as fallback
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto" if not dist.is_initialized() else {"": local_rank}
        )
    
    # Prepare the model for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA config
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if local_rank == 0:
        print(f"Using target modules for LoRA: {target_modules}")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules
    )
    
    # Get the PEFT model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters info only on main process
    if local_rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of total {total_params:,})")
    
    return model, tokenizer

# Function to tokenize the dataset
def tokenize_dataset(formatted_data, tokenizer):
    def tokenize_function(examples):
        # Combine system, input, and output for causal language modeling
        full_prompts = []
        for i in range(len(examples["system"])):
            # Format according to Qwen2.5 chat template
            chat = [
                {"role": "system", "content": examples["system"][i]},
                {"role": "user", "content": examples["input"][i]},
                {"role": "assistant", "content": examples["output"][i]}
            ]
            full_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            full_prompts.append(full_prompt)
        
        # Tokenize
        tokenized = tokenizer(
            full_prompts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Set labels equal to input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Convert formatted data to the format expected by the tokenize function
    dataset_dict = {
        "system": [item["system"] for item in formatted_data],
        "input": [item["input"] for item in formatted_data],
        "output": [item["output"] for item in formatted_data]
    }
    
    # Create dataset from dictionary
    from datasets import Dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

# Main fine-tuning function
def fine_tune(args=None):
    # Set up distributed training
    local_rank, is_main_process = setup_distributed()
    
    # Override configuration with command-line arguments if provided
    global OUTPUT_DIR, BATCH_SIZE, NUM_EPOCHS
    if args is not None:
        if args.output_dir:
            OUTPUT_DIR = args.output_dir
        if args.batch_size:
            BATCH_SIZE = args.batch_size
        if args.num_epochs:
            NUM_EPOCHS = args.num_epochs
    
    try:
        # Prepare the dataset
        formatted_data = prepare_dataset(is_main_process)
        
        # Check if we have any data
        if not formatted_data and is_main_process:
            print("No data formatted successfully. Please check dataset format.")
            return
        
        # Clear GPU memory before preparing the model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Prepare the model and tokenizer
        model, tokenizer = prepare_model(local_rank)
        
        # Tokenize the dataset
        tokenized_dataset = tokenize_dataset(formatted_data, tokenizer)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Define training arguments without DeepSpeed
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            fp16=FP16,
            bf16=False,
            save_total_limit=3,
            remove_unused_columns=False,
            report_to=["none"],  # Disable all reporting tools
            # Add gradient checkpointing to save memory
            gradient_checkpointing=True,
            seed=SEED,  # Set seed for reproducibility
            # Distributed training parameters
            local_rank=local_rank,
            ddp_find_unused_parameters=False,  # Improve performance, since we know all params are used
            ddp_bucket_cap_mb=25,  # Adjust DDP communication bucket size
            dataloader_num_workers=4,  # Use multiple workers for faster data loading
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        # Train the model
        if is_main_process:
            print("Starting fine-tuning...")
        trainer.train()
        
        # Save the final model - only main process saves
        if is_main_process:
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Fine-tuning complete. Model saved to {OUTPUT_DIR}")
            
            # Also save training configuration for reproducibility
            with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
                config = {
                    "model_name": MODEL_NAME,
                    "dataset_name": DATASET_NAME,
                    "train_ratio": TRAIN_RATIO,
                    "test_use_ratio": TEST_USE_RATIO,
                    "num_samples": NUM_SAMPLES,
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "lora_dropout": LORA_DROPOUT,
                    "learning_rate": LEARNING_RATE,
                    "max_length": MAX_LENGTH,
                    "batch_size": BATCH_SIZE,
                    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                    "num_epochs": NUM_EPOCHS,
                    "fp16": FP16,
                    "seed": SEED,
                    "num_gpus": torch.cuda.device_count()
                }
                json.dump(config, f, indent=2)
    
    except Exception as e:
        if is_main_process:
            print(f"Error during fine-tuning: {e}")
            
            # If we encounter an OOM error, try with an even smaller batch or more aggressive memory saving
            if "CUDA out of memory" in str(e):
                print("Out of memory error. Consider these options:")
                print("1. Further reduce batch size or sequence length")
                print("2. Use fewer examples (reduce NUM_SAMPLES)")
                print("3. Use gradient checkpointing to save memory")
                print("4. Clear more GPU memory before training")
    
    # Make sure to clean up by destroying process group
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Run the fine-tuning process
    fine_tune()