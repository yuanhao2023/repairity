#!/usr/bin/env python3
# train_reward_model.py - Train a binary reward model for code quality

import os
import json
import torch
import argparse
import logging
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define compute_metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Train a binary reward model for code quality")
    parser.add_argument("--model_name", default="codellama/CodeLlama-7b-hf", 
                      help="Base model to use (default: codellama/CodeLlama-7b-hf)")
    parser.add_argument("--feedback_file", default="./data/binary_feedback.json",
                      help="Path to binary feedback JSON file (default: ./data/binary_feedback.json)")
    parser.add_argument("--output_dir", default="./models/binary_reward_model",
                      help="Output directory for trained model")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024,
                      help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true",
                      help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true",
                      help="Load model in 8-bit quantization")
    parser.add_argument("--lora_r", type=int, default=16,
                      help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                      help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                      help="LoRA dropout value")
    return parser.parse_args()

def prepare_dataset(data_path, tokenizer, max_length):
    """Load and prepare the binary feedback dataset for training."""
    logger.info(f"Loading binary feedback data from {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Convert to binary labels
    examples = []
    for item in data:
        label = 1 if item["label"] == "GOOD" else 0
        examples.append({
            "input": item["input"],
            "solution": item["solution"],
            "label": label
        })
    
    # Create dataset
    dataset = Dataset.from_list(examples)
    
    # Split into train/test
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    
    def tokenize_function(examples):
        # Combine input and solution for classification
        texts = [f"Task: {inp}\n\nSolution: {sol}" for inp, sol in 
                zip(examples["input"], examples["solution"])]
        
        tokenized = tokenizer(texts, padding="max_length", truncation=True, 
                        max_length=max_length)
        
        return tokenized
    
    # Tokenize datasets
    tokenized_train = train_test["train"].map(
        tokenize_function, 
        batched=True,
        remove_columns=["input", "solution"]
    )
    
    tokenized_test = train_test["test"].map(
        tokenize_function, 
        batched=True,
        remove_columns=["input", "solution"]
    )
    
    logger.info(f"Prepared dataset with {len(tokenized_train)} training examples and {len(tokenized_test)} test examples")
    
    # Check distribution of labels in train set
    train_labels = train_test["train"]["label"]
    good_count = sum(train_labels)
    bad_count = len(train_labels) - good_count
    
    logger.info(f"Training set has {good_count} GOOD examples ({good_count / len(train_labels):.2%}) and {bad_count} BAD examples")
    
    return tokenized_train, tokenized_test

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up file logger
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "training.log"))
    logger.addHandler(file_handler)
    
    # Log all arguments
    logger.info(f"Training with arguments: {args}")
    
    # Set up quantization config if needed
    quantization_config = None
    use_peft = False
    
    if args.load_in_4bit:
        logger.info("Using 4-bit quantization with LoRA adapters")
        use_peft = True
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif args.load_in_8bit:
        logger.info("Using 8-bit quantization with LoRA adapters")
        use_peft = True
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    try:
        logger.info(f"Loading tokenizer for model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            else:
                logger.warning("No pad_token or eos_token found. Using default padding token.")
        
        # Load model for sequence classification (binary)
        logger.info(f"Loading model: {args.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Apply LoRA adapters if using quantization
        if use_peft:
            logger.info("Preparing model for k-bit training")
            model = prepare_model_for_kbit_training(model)
            
            logger.info(f"Adding LoRA adapters with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    except OSError as e:
        if "403 Forbidden" in str(e) or "401 Unauthorized" in str(e):
            logger.error(f"ERROR: Access to model {args.model_name} is restricted.")
            logger.error("Some models like google/codegemma-7b-it require access approval from HuggingFace.")
            logger.error("Please try one of these alternatives:")
            logger.error("  - codellama/CodeLlama-7b-hf")
            logger.error("  - Salesforce/codegen-350M-mono")
            logger.error("  - Salesforce/codegen-2B-mono")
            logger.error("  - google/codegemma-2b (if available)")
            logger.error("\nOr visit the model page to request access and accept terms of use.")
            return
        else:
            # Re-raise the exception for other issues
            raise
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(
        args.feedback_file, tokenizer, args.max_length
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        seed=42
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate final model
    final_eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {final_eval_results}")
    
    # Save model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")
    
    with open(f"{args.output_dir}/eval_results.json", "w") as f:
        json.dump(final_eval_results, f, indent=2)
    
    # Create a simple model card with usage instructions
    with open(f"{args.output_dir}/README.md", "w") as f:
        f.write(f"""# Binary Code Quality Reward Model

This model was fine-tuned from {args.model_name} to classify code solutions as either GOOD or BAD.

## Training Information
- Base Model: {args.model_name}
- Training Data: Binary feedback from Claude 3.7
- Epochs: {args.epochs}
- Learning Rate: {args.learning_rate}
- PEFT Method: {"LoRA" if use_peft else "None"}
- Quantization: {"4-bit" if args.load_in_4bit else "8-bit" if args.load_in_8bit else "None"}
- Final Evaluation Results: {final_eval_results}

## Usage Example

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

# For regular models
model_path = "{args.output_dir}"

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Function to get quality score
def get_quality_score(task, solution):
    # Combine task and solution
    input_text = f"Task: {{task}}\\n\\nSolution: {{solution}}"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length={args.max_length})
    
    # Get model prediction
    import torch
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        
    # Return probability of "GOOD" class (index 1)
    good_score = probabilities[0][1].item()
    return good_score
```

## License
This model inherits the license of the base model ({args.model_name}).
""")

def create_wrapper_script(args):
    """Create a convenient wrapper script for using the reward model."""
    script_path = os.path.join(args.output_dir, "predict_quality.py")
    with open(script_path, "w") as f:
        f.write("""#!/usr/bin/env python3
# predict_quality.py - Script to predict code quality using the reward model

import os
import torch
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Predict code quality score")
    parser.add_argument("--task", required=True, help="Programming task description or path to task file")
    parser.add_argument("--solution", required=True, help="Code solution or path to solution file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load task and solution from files if paths are provided
    task = args.task
    solution = args.solution
    
    if os.path.exists(args.task):
        with open(args.task, 'r') as f:
            task = f.read()
            
    if os.path.exists(args.solution):
        with open(args.solution, 'r') as f:
            solution = f.read()
    
    # Load model and tokenizer from the same directory as this script
    model_path = os.path.dirname(os.path.abspath(__file__))
    
    # Try to load as a PEFT model first
    try:
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, 
            num_labels=2,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, model_path)
        print(f"Loaded PEFT model based on {config.base_model_name_or_path}")
    except Exception:
        # Fall back to regular model loading
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("Loaded regular (non-PEFT) model")
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Combine task and solution
    input_text = f"Task: {task}\\n\\nSolution: {solution}"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    
    # Get scores
    bad_score = probabilities[0][0].item()
    good_score = probabilities[0][1].item()
    
    # Determine verdict
    verdict = "GOOD" if good_score > 0.5 else "BAD"
    
    # Print results
    print(f"Quality Scores:")
    print(f"  GOOD: {good_score:.4f} ({good_score * 100:.1f}%)")
    print(f"  BAD:  {bad_score:.4f} ({bad_score * 100:.1f}%)")
    print(f"Verdict: {verdict}")

if __name__ == "__main__":
    main()
""")
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    logger.info(f"Created executable wrapper script at {script_path}")

if __name__ == "__main__":
    main()
    create_wrapper_script(parse_args())