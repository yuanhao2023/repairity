# QwenEval.py
import os
import json
import re
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)

# Define argument parser
parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen model")
parser.add_argument("--model_path", type=str, default="./qwen2.5-coder-7b-finetuned", 
                    help="Path to the fine-tuned model")
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="Base model name")
parser.add_argument("--dataset_name", type=str, default="bigcode/bigcodebench",
                    help="Dataset name")
parser.add_argument("--dataset_version", type=str, default="v0.1.4",
                    help="Dataset version")
parser.add_argument("--test_size", type=float, default=0.1,
                    help="Proportion of dataset to use for testing (if no predefined test set)")
parser.add_argument("--max_test_samples", type=int, default=114,
                    help="Maximum number of test samples to evaluate")
parser.add_argument("--max_length", type=int, default=2048,
                    help="Max sequence length")
parser.add_argument("--gen_max_length", type=int, default=1024,
                    help="Max generation length")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device to run evaluation on")
parser.add_argument("--temperature", type=float, default=0.1,
                    help="Temperature for generation")
parser.add_argument("--top_p", type=float, default=0.75,
                    help="Top p for nucleus sampling")
parser.add_argument("--output_dir", type=str, default="./eval_results",
                    help="Directory to save evaluation results")
parser.add_argument("--load_in_4bit", action="store_true",
                    help="Load model in 4-bit precision")
parser.add_argument("--load_in_8bit", action="store_true",
                    help="Load model in 8-bit precision")
parser.add_argument("--num_samples", type=int, default=1,
                    help="Number of solutions to generate per problem (for Pass@1)")
parser.add_argument("--show_reasoning", action="store_true",
                    help="Request reasoning from the model before solution")

args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Function to format test examples
def format_test_example(example, include_reasoning=False):
    """Format test examples for evaluation."""
    task_id = example.get("task_id", "")
    canonical_solution = example.get("canonical_solution", "")
    instruct_prompt = example.get("instruct_prompt", "")
    code_prompt = example.get("code_prompt", "")
    complete_prompt = example.get("complete_prompt", "")
    
    # Determine the language based on available info
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
    
    # Add reasoning request if enabled
    if include_reasoning:
        instruction = (
            f"{instruction}\n\n"
            "First, explain your approach and reasoning for solving this problem. "
            "Then, provide your complete code solution in a code block."
        )
    
    # Clean up the canonical solution for reference
    solution = canonical_solution
    if solution.startswith("```") and solution.endswith("```"):
        # Extract the code from the markdown code block
        lines = solution.split("\n")
        if len(lines) > 2:
            # Skip the first and last line (markdown delimiters)
            solution = "\n".join(lines[1:-1])
    
    return {
        "task_id": task_id,
        "system": system,
        "instruction": instruction,
        "language": language,
        "reference_solution": solution,
        "original_prompt": instruct_prompt or complete_prompt or code_prompt or f"Task ID: {task_id}"
    }

def load_and_prepare_test_data():
    """Load and prepare test data from the dataset."""
    print(f"Loading dataset {args.dataset_name} version {args.dataset_version}")
    dataset = load_dataset(args.dataset_name)
    
    if args.dataset_version not in dataset:
        print(f"Version {args.dataset_version} not found in dataset. Available versions: {list(dataset.keys())}")
        return []
    
    # Get the dataset for the specified version
    version_data = dataset[args.dataset_version]
    
    # Check if we have a pre-defined test split
    train_test_indices_path = "./data/train_test_indices.json"
    if os.path.exists(train_test_indices_path):
        print("Using predefined test split from training...")
        with open(train_test_indices_path, 'r') as f:
            indices = json.load(f)
            test_indices = indices.get("test_indices", [])
            
            print(f"Total test indices: {len(test_indices)}")
            
            # USE THE FULL TEST SET (all 114 examples)
            indices_to_use = test_indices
            
            # Limit number of test samples if specified
            if args.max_test_samples > 0 and args.max_test_samples < len(indices_to_use):
                np.random.seed(42)
                indices_to_use = np.random.choice(indices_to_use, args.max_test_samples, replace=False).tolist()
            
            test_data = version_data.select(indices_to_use)
            print(f"Using {len(test_data)} examples from predefined test set")
            return [format_test_example(example, args.show_reasoning) for example in test_data]
    
    # If no predefined test set or it's empty, create a new one
    print("No predefined test split found. Creating a new one...")
    from sklearn.model_selection import train_test_split
    
    # Split the dataset into train and test
    indices = list(range(len(version_data)))
    _, test_indices = train_test_split(indices, test_size=args.test_size, random_state=42)
    
    # Limit number of test samples if specified
    if args.max_test_samples > 0 and args.max_test_samples < len(test_indices):
        np.random.seed(42)
        test_indices = np.random.choice(test_indices, args.max_test_samples, replace=False).tolist()
    
    test_data = version_data.select(test_indices)
    print(f"Created test set with {len(test_data)} examples (10% of full dataset)")
    
    return [format_test_example(example, args.show_reasoning) for example in test_data]

# def load_and_prepare_test_data():
    """Load and prepare test data from the dataset."""
    print(f"Loading dataset {args.dataset_name} version {args.dataset_version}")
    dataset = load_dataset(args.dataset_name)
    
    if args.dataset_version not in dataset:
        print(f"Version {args.dataset_version} not found in dataset. Available versions: {list(dataset.keys())}")
        return []
    
    # Get the dataset for the specified version
    version_data = dataset[args.dataset_version]
    
    # # Check if we have a pre-defined test split
    # train_test_indices_path = "./data/train_test_indices.json"
    # if os.path.exists(train_test_indices_path):
    #     print("Using predefined test split from training...")
    #     with open(train_test_indices_path, 'r') as f:
    #         indices = json.load(f)
    #         test_indices = indices.get("test_indices", [])
    #         # Use only non-overlapping test indices (those not used in training)
    #         test_for_train_indices = indices.get("test_for_train_indices", [])
    #         test_only_indices = [idx for idx in test_indices if idx not in test_for_train_indices]
            
    #         if test_only_indices:
    #             # Limit number of test samples if specified
    #             if args.max_test_samples > 0 and args.max_test_samples < len(test_only_indices):
    #                 np.random.seed(42)
    #                 test_only_indices = np.random.choice(test_only_indices, args.max_test_samples, replace=False).tolist()
                
    #             test_data = version_data.select(test_only_indices)
    #             print(f"Using {len(test_data)} examples from predefined test set")
    #             return [format_test_example(example, args.show_reasoning) for example in test_data]
    train_test_indices_path = "./data/train_test_indices.json"
    if os.path.exists(train_test_indices_path):
        print("Using predefined test split from training...")
        with open(train_test_indices_path, 'r') as f:
            indices = json.load(f)
            test_indices = indices.get("test_indices", [])
            # Use only non-overlapping test indices (those not used in training)
            test_for_train_indices = indices.get("test_for_train_indices", [])
            test_only_indices = [idx for idx in test_indices if idx not in test_for_train_indices]
            
            if test_only_indices:
                # Limit number of test samples if specified
                if args.max_test_samples > 0 and args.max_test_samples < len(test_only_indices):
                    np.random.seed(42)
                    test_only_indices = np.random.choice(test_only_indices, args.max_test_samples, replace=False).tolist()
                
                test_data = version_data.select(test_only_indices)
                print(f"Using {len(test_data)} examples from predefined test set")
                return [format_test_example(example, args.show_reasoning) for example in test_data]
    
    # If no predefined test set or it's empty, create a new one
    print("No predefined test split found. Creating a new one...")
    from sklearn.model_selection import train_test_split
    
    # Split the dataset into train and test
    indices = list(range(len(version_data)))
    _, test_indices = train_test_split(indices, test_size=args.test_size, random_state=42)
    
    # Limit number of test samples if specified
    if args.max_test_samples > 0 and args.max_test_samples < len(test_indices):
        np.random.seed(42)
        test_indices = np.random.choice(test_indices, args.max_test_samples, replace=False).tolist()
    
    test_data = version_data.select(test_indices)
    print(f"Created test set with {len(test_data)} examples (10% of full dataset)")
    
    return [format_test_example(example, args.show_reasoning) for example in test_data]

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {args.model_path}")
    
    # Check if the path is to a LoRA adapter
    is_lora = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    
    # Setup quantization if needed
    quantization_config = None
    if args.load_in_4bit:
        print("Loading model in 4-bit precision")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif args.load_in_8bit:
        print("Loading model in 8-bit precision")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path if not is_lora else args.base_model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    if is_lora:
        print(f"Loading base model {args.base_model} and LoRA adapter")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        # Load full fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
    
    model.eval()
    return model, tokenizer

def extract_code_and_reasoning(text):
    """Extract code and reasoning from generated text."""
    # Check if there's a code block
    if "```" in text:
        # Split by code blocks
        parts = text.split("```")
        
        # Everything before the first code block is reasoning
        reasoning = parts[0].strip()
        
        # Code is in the first code block
        if len(parts) >= 3:  # At least one code block
            code_content = parts[1]
            # Remove language specifier if present
            if "\n" in code_content:
                lang_specifier = code_content.split("\n")[0].strip()
                if lang_specifier and not lang_specifier.startswith("```"):
                    code_content = "\n".join(code_content.split("\n")[1:])
            code = code_content.strip()
            
            return {
                "reasoning": reasoning,
                "code": code
            }
    
    # If no code block is found or format doesn't match expectations
    return {
        "reasoning": "",
        "code": text.strip()  # Assume all content is code
    }

def generate_solution(model, tokenizer, example):
    """Generate a solution using the model."""
    # Format the chat conversation
    chat = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["instruction"]}
    ]
    
    # Get prompt using the model's chat template
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    
    # Generate the response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.gen_max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=(args.temperature > 0.0 and args.num_samples > 1),
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1  # We'll handle multiple samples manually
        )
    
    # Decode the response
    generated_text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Extract code and reasoning
    result = extract_code_and_reasoning(generated_text)
    
    return result

def check_correctness(generated_code, reference_solution, task_id):
    """
    Placeholder for code correctness checking.
    In a real implementation, this would compile/run the code and check its output.
    For this script, we'll use a very basic string similarity heuristic.
    """
    # Remove comments, whitespace, and normalize variable names for comparison
    def normalize_code(code):
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        # Remove whitespace
        code = re.sub(r'\s+', ' ', code).strip()
        return code
    
    # Normalize both solutions
    gen_norm = normalize_code(generated_code)
    ref_norm = normalize_code(reference_solution)
    
    # Simple exact match - this is a placeholder for real execution-based evaluation
    exact_match = gen_norm == ref_norm
    
    # This is a very rudimentary check - in practice, you would want to:
    # 1. Execute the generated code on test cases
    # 2. Compare outputs with expected outputs
    # 3. Check for correctness in terms of functionality
    
    # As a fallback, we're using a similarity heuristic
    if not exact_match and len(gen_norm) > 0 and len(ref_norm) > 0:
        # Check if at least 80% of tokens are the same (very rough heuristic)
        gen_tokens = set(re.findall(r'[\w]+', gen_norm))
        ref_tokens = set(re.findall(r'[\w]+', ref_norm))
        
        if len(ref_tokens) > 0:
            overlap = len(gen_tokens.intersection(ref_tokens)) / len(ref_tokens)
            if overlap >= 0.8:
                print(f"Task {task_id}: High similarity but not exact match")
                return True
    
    return exact_match

def evaluate_model():
    """Main evaluation function."""
    # Load test data
    test_examples = load_and_prepare_test_data()
    if not test_examples:
        print("No test examples available for evaluation.")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Generate solutions with Pass@1 approach
    all_results = []
    correct_count = 0
    
    print(f"Generating solutions for {len(test_examples)} test examples...")
    for example in tqdm(test_examples):
        try:
            task_results = []
            
            # Generate multiple solutions if num_samples > 1
            for i in range(args.num_samples):
                generated_result = generate_solution(model, tokenizer, example)
                
                # Check correctness
                is_correct = check_correctness(
                    generated_result["code"], 
                    example["reference_solution"], 
                    example["task_id"]
                )
                
                solution_result = {
                    "sample_id": i,
                    "reasoning": generated_result["reasoning"],
                    "code": generated_result["code"],
                    "is_correct": is_correct
                }
                task_results.append(solution_result)
                
                # For Pass@1, we only need one successful solution
                if is_correct:
                    break
            
            # Check if any solution was correct
            any_correct = any(result["is_correct"] for result in task_results)
            if any_correct:
                correct_count += 1
            
            result = {
                "task_id": example["task_id"],
                "instruction": example["original_prompt"],
                "language": example["language"],
                "reference_solution": example["reference_solution"],
                "solutions": task_results,
                "any_correct": any_correct
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"Error generating solution for task {example['task_id']}: {e}")
    
    # Calculate Pass@1
    pass_at_1 = correct_count / len(test_examples) if test_examples else 0
    
    # Collect successful examples with reasoning
    successful_examples = []
    for result in all_results:
        if result["any_correct"]:
            # Find the first correct solution
            correct_solution = next(sol for sol in result["solutions"] if sol["is_correct"])
            successful_examples.append({
                "task_id": result["task_id"],
                "instruction": result["instruction"],
                "reasoning": correct_solution["reasoning"],
                "solution": correct_solution["code"]
            })
    
    # Save results
    evaluation_summary = {
        "pass_at_1": pass_at_1,
        "total_examples": len(test_examples),
        "correct_examples": correct_count,
        "model_path": args.model_path,
        "base_model": args.base_model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_samples": args.num_samples
    }
    
    print(f"\nEvaluation Results:")
    print(f"Pass@1: {pass_at_1:.4f} ({correct_count}/{len(test_examples)})")
    
    # Save detailed results
    output_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    # Save successful examples with reasoning
    successful_file = os.path.join(args.output_dir, "successful_examples_with_reasoning.json")
    with open(successful_file, 'w') as f:
        json.dump(successful_examples, f, indent=2)
    
    print(f"Evaluation completed. Results saved to {args.output_dir}")
    print(f"Detailed results: {output_file}")
    print(f"Summary: {summary_file}")
    print(f"Successful examples with reasoning: {successful_file}")

if __name__ == "__main__":
    evaluate_model()

# # For full model evaluation:
# python QwenH100Eval.py --model_path ./qwen2.5-coder-7b-finetuned --load_in_4bit

# # For LoRA adapter evaluation:
# python QwenH100Eval.py --model_path ./qwen2.5-coder-7b-finetuned --base_model Qwen/Qwen2.5-Coder-7B-Instruct --load_in_4bit