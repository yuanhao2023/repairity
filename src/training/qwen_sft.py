# QwenSFT.py
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
OUTPUT_DIR = "./qwen2.5-coder-7b-finetuned"
DATASET_NAME = "bigcode/bigcodebench"
DATASET_VERSION = "v0.1.4"  # Using v0.1.4 as specified
TRAIN_RATIO = 0.9  # 90% of data for training
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

# Training Strategy Options
REASONING_STRATEGY = "explicit"  # Options: "explicit", "cot", "alternating"

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

# Function to format BigCodeBench items for instruction fine-tuning
def format_instruction(example):
    """Format BigCodeBench examples for instruction tuning."""
    
    # Extract relevant fields from the BigCodeBench structure
    task_id = example.get("task_id", "")
    canonical_solution = example.get("canonical_solution", "")
    instruct_prompt = example.get("instruct_prompt", "")
    code_prompt = example.get("code_prompt", "")
    complete_prompt = example.get("complete_prompt", "")
    
    # Determine the language based on available info
    # Default to python if not specified
    language = "python"
    if isinstance(canonical_solution, str) and canonical_solution.startswith("```"):
        # Try to extract language from markdown code block
        lang_line = canonical_solution.split("\n")[0]
        if lang_line.startswith("```"):
            lang = lang_line.replace("```", "").strip()
            if lang:
                language = lang
    
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
    
    # Clean up the canonical solution if needed
    solution = canonical_solution
    if solution.startswith("```") and solution.endswith("```"):
        # Extract the code from the markdown code block
        lines = solution.split("\n")
        if len(lines) > 2:
            # Skip the first and last line (markdown delimiters)
            solution = "\n".join(lines[1:-1])
    
    # Create formatted input and output
    formatted_input = instruction
    formatted_output = f"```{language}\n{solution}\n```"
    
    return {
        "system": system,
        "input": formatted_input,
        "output": formatted_output
    }

def format_claude_reasoning_example(item, strategy="explicit"):
    """Format Claude reasoning traces with different strategies for SFT."""
    
    # Extract relevant fields
    task = item.get("task", "")
    reasoning_trace = item.get("reasoning_trace", "")
    solution = item.get("solution", "")
    sections = item.get("sections", {})
    
    # Determine language (default to python)
    language = "python"
    
    # Extract reasoning components
    approach = sections.get("reasoning", "")
    solution_code = sections.get("solution", solution)
    
    # Different strategies for incorporating reasoning
    if strategy == "explicit":
        # Explicit reasoning structure
        system = "You are a skilled programmer. When solving coding problems, always explain your approach first, then provide your solution code."
        formatted_input = task
        formatted_output = f"## APPROACH\n{approach.strip()}\n\n## CODE\n```{language}\n{solution_code}\n```"
    
    elif strategy == "cot":
        # Chain-of-thought style
        system = "You are a skilled programmer. Solve problems step-by-step, explaining your reasoning as you develop the solution."
        formatted_input = f"Solve this coding problem step by step:\n\n{task}"
        
        # Create a more structured chain of thought
        steps = []
        if "analysis" in sections:
            steps.append(f"First, I'll analyze the problem: {sections['analysis']}")
        if "issue_identification" in sections:
            steps.append(f"Key considerations: {sections['issue_identification']}")
        
        # Add the main reasoning
        steps.append(f"My approach: {approach}")
        
        # Create CoT output
        cot_text = "\n\n".join(steps)
        formatted_output = f"{cot_text}\n\nFinal solution:\n```{language}\n{solution_code}\n```"
    
    elif strategy == "alternating":
        # Break solution into parts with alternating reasoning
        system = "You are a skilled programmer who explains code step by step."
        formatted_input = task
        
        # Try to match the solution structure from the approach
        code_parts = solution_code.split("\n\n")
        if len(code_parts) <= 1:
            # If can't split meaningfully, just use the explicit approach
            formatted_output = f"## APPROACH\n{approach.strip()}\n\n## CODE\n```{language}\n{solution_code}\n```"
        else:
            # Create alternating reasoning and code parts
            formatted_parts = []
            formatted_parts.append(f"First, I'll consider my overall approach:\n{approach.strip()}")
            
            for i, part in enumerate(code_parts):
                if part.strip():
                    formatted_parts.append(f"Part {i+1} of the solution:\n```{language}\n{part.strip()}\n```")
                    
            formatted_output = "\n\n".join(formatted_parts)
    
    else:
        # Default to just the solution (no reasoning)
        system = "You are a skilled programmer. Provide efficient and correct code solutions."
        formatted_input = task
        formatted_output = f"```{language}\n{solution_code}\n```"
    
    return {
        "system": system,
        "input": formatted_input,
        "output": formatted_output
    }

# Load and prepare the dataset
def prepare_dataset(is_main_process, claude_data_path=None, reasoning_strategy=REASONING_STRATEGY):
    # First, check if we should use Claude reasoning traces
    if claude_data_path and os.path.exists(claude_data_path):
        if is_main_process:
            print(f"Loading Claude reasoning traces from {claude_data_path}...")
            print(f"Using reasoning strategy: {reasoning_strategy}")
        
        with open(claude_data_path, 'r') as f:
            claude_data = json.load(f)
        
        if is_main_process:
            print(f"Loaded {len(claude_data)} examples with Claude reasoning traces")
        
        # Split into train/test (90/10)
        random.seed(SEED)
        train_size = int(len(claude_data) * TRAIN_RATIO)
        train_data = claude_data[:train_size]
        test_data = claude_data[train_size:]
        
        if is_main_process:
            print(f"Split into {len(train_data)} training examples and {len(test_data)} test examples")
            # Save test data for evaluation
            os.makedirs("./data/test", exist_ok=True)
            with open(f"./data/test/claude_traces_test_set_{reasoning_strategy}.json", "w") as f:
                json.dump(test_data, f, indent=2)
        
        # Format the examples for instruction tuning with reasoning traces
        formatted_data = []
        success_count = 0
        
        for item in train_data:
            try:
                # Extract relevant fields
                reasoning_trace = item.get("reasoning_trace", "")
                solution = item.get("solution", "")
                
                # Skip if no reasoning trace or solution
                if not reasoning_trace or not solution:
                    if is_main_process:
                        print(f"Skipping example {item.get('task_id', 'unknown')} due to missing reasoning trace or solution")
                    continue
                
                # Format the example with chosen reasoning strategy
                formatted_example = format_claude_reasoning_example(item, strategy=reasoning_strategy)
                formatted_data.append(formatted_example)
                success_count += 1
                
            except Exception as e:
                if is_main_process:
                    print(f"Error formatting Claude example {item.get('task_id', 'unknown')}: {e}")
                continue
        
        if is_main_process:
            print(f"Successfully formatted {success_count} Claude examples for training")
            
            # Save a few examples for inspection
            os.makedirs("./data/training_examples", exist_ok=True)
            with open(f"./data/training_examples/claude_formatted_{reasoning_strategy}.json", "w") as f:
                json.dump(formatted_data[:5], f, indent=2)
        
        return formatted_data
    
    # If not using Claude data, proceed with the original BigCodeBench dataset
    if is_main_process:
        print(f"Loading BigCodeBench dataset (version: {DATASET_VERSION})...")
    
    # Load the dataset
    dataset = load_dataset(DATASET_NAME)
    
    # Check if the requested version exists
    if DATASET_VERSION not in dataset:
        if is_main_process:
            print(f"Version {DATASET_VERSION} not found in dataset. Available versions: {list(dataset.keys())}")
        return []
    
    # Get the dataset for the specified version
    version_data = dataset[DATASET_VERSION]
    
    if is_main_process:
        print(f"Dataset size: {len(version_data)}")
    
    # Set random seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Split the data into train and test sets (90% train, 10% test)
    train_indices, test_indices = train_test_split(
        range(len(version_data)), 
        train_size=TRAIN_RATIO, 
        random_state=SEED
    )
    
    # Create the final training dataset - using ONLY training indices
    train_data = version_data.select(train_indices)
    
    if is_main_process:
        print(f"Training data size: {len(train_data)} examples")
        print(f"Test data size: {len(test_indices)} examples (kept separate for evaluation)")
    
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
        print("CUDA is available, using GPU")
    else:
        print("CUDA is not available, using CPU mode")
    
    try:
        # Only try 4-bit quantization if CUDA is available
        if torch.cuda.is_available():
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
        else:
            raise ValueError("CUDA not available, using CPU fallback")
        
    except Exception as e:
        if local_rank == 0:
            print(f"Failed to load with 4-bit quantization: {e}")
            print("Falling back to standard loading...")
        
        # Load model without quantization as fallback
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if not dist.is_initialized() else {"": local_rank}
        )
    
    try:
        # Prepare the model for LoRA fine-tuning
        if torch.cuda.is_available():
            model = prepare_model_for_kbit_training(model)
    except Exception as e:
        if local_rank == 0:
            print(f"Error in prepare_model_for_kbit_training: {e}")
            print("Continuing without kbit preparation...")
    
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
    
    # Fix some parameters to ensure gradients flow
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
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
    global OUTPUT_DIR, BATCH_SIZE, NUM_EPOCHS, REASONING_STRATEGY
    claude_data_path = None
    
    if args is not None:
        if args.output_dir:
            OUTPUT_DIR = args.output_dir
        if args.batch_size:
            BATCH_SIZE = args.batch_size
        if args.num_epochs:
            NUM_EPOCHS = args.num_epochs
        if hasattr(args, 'claude_data_path'):
            claude_data_path = args.claude_data_path
        if hasattr(args, 'reasoning_strategy'):
            REASONING_STRATEGY = args.reasoning_strategy
    
    try:
        # Prepare the dataset with Claude data and chosen reasoning strategy
        formatted_data = prepare_dataset(is_main_process, claude_data_path, REASONING_STRATEGY)
        
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
        
        # Output directory with reasoning strategy
        model_output_dir = f"{OUTPUT_DIR}_{REASONING_STRATEGY}"
        
        # Define training arguments without DeepSpeed
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            fp16=FP16 and torch.cuda.is_available(),  # Only use fp16 if CUDA is available
            bf16=False,
            save_total_limit=3,
            remove_unused_columns=False,
            report_to=["none"],  # Disable all reporting tools
            # Add gradient checkpointing to save memory, but only if CUDA is available
            gradient_checkpointing=torch.cuda.is_available(),
            seed=SEED,  # Set seed for reproducibility
            # Distributed training parameters
            local_rank=local_rank,
            ddp_find_unused_parameters=False,  # Improve performance, since we know all params are used
            ddp_bucket_cap_mb=25,  # Adjust DDP communication bucket size
            dataloader_num_workers=4 if torch.cuda.is_available() else 1,  # Use multiple workers for faster data loading
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
            print(f"Starting fine-tuning with {REASONING_STRATEGY} reasoning strategy...")
        trainer.train()
        
        # Save the final model - only main process saves
        if is_main_process:
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
            print(f"Fine-tuning complete. Model saved to {model_output_dir}")
            
            # Also save training configuration for reproducibility
            with open(os.path.join(model_output_dir, "training_config.json"), "w") as f:
                config = {
                    "model_name": MODEL_NAME,
                    "dataset_name": DATASET_NAME,
                    "dataset_version": DATASET_VERSION,
                    "train_ratio": TRAIN_RATIO,
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
                    "num_gpus": torch.cuda.device_count(),
                    "reasoning_strategy": REASONING_STRATEGY
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