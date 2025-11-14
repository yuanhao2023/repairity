#!/usr/bin/env python3
# run_eval.py - Direct evaluation script for TrustCode models

import os
import sys
import argparse
import subprocess
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run evaluation for TrustCode models")
    parser.add_argument("--model_path", required=True, 
                       help="Path to the model to evaluate")
    parser.add_argument("--output_dir", default=None,
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Extract strategy from model path
    model_parts = args.model_path.split('_')
    strategy = model_parts[-1] if len(model_parts) > 1 else "unknown"
    
    # Create timestamped output directory if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./eval_results_{strategy}_{timestamp}"
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running evaluation for model: {args.model_path}")
    print(f"Results will be saved to: {args.output_dir}")
    
    # Try multiple command forms until one works
    commands = [
        f"python -m src.evaluation.qwen_eval --model_path {args.model_path} --output_dir {args.output_dir}",
        f"python -m src.main evaluate --model_path {args.model_path} --output_dir {args.output_dir}",
        f"python -m src.main evaluate --model qwen-7b --model_path {args.model_path} --output_dir {args.output_dir}",
        f"python -m src.main --model_path {args.model_path} --output_dir {args.output_dir}"
    ]
    
    success = False
    for cmd in commands:
        print(f"\nTrying command: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
            success = True
            print(f"Evaluation successful with command: {cmd}")
            break
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {e}")
    
    if not success:
        print("\nAll evaluation commands failed. Please try the following fallback approach:")
        print(f"1. Check src/evaluation/qwen_eval.py for the correct function signature")
        print(f"2. Run evaluation directly: python -m src.evaluation.qwen_eval --model_path {args.model_path}")
        sys.exit(1)
    
    print(f"\nEvaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 