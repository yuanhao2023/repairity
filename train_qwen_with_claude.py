#!/usr/bin/env python3
# train_qwen_with_claude.py - Script to run training with Claude reasoning traces

import os
import argparse
import subprocess
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen with Claude reasoning traces")
    parser.add_argument("--claude_data_path", default="./data/claude_reasoning_traces_v0.1.4_100.json",
                       help="Path to Claude reasoning traces JSON file")
    parser.add_argument("--strategies", nargs="+", default=["explicit"], 
                       choices=["explicit", "cot", "alternating", "none"],
                       help="Reasoning strategies to use for training")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per GPU")
    parser.add_argument("--output_dir", default=None,
                       help="Base output directory for models")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                       help="Ratio of data to use for training (rest for test)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create timestamped output directory if not provided
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./models/qwen_repairity_{timestamp}"
    
    print(f"Training Qwen model with Claude reasoning traces")
    print(f"Using data from: {args.claude_data_path}")
    print(f"Training with strategies: {args.strategies}")
    print(f"Train/test split: {args.train_ratio:.1%} training, {1-args.train_ratio:.1%} testing")
    
    # Ensure the Claude data file exists
    if not os.path.exists(args.claude_data_path):
        raise FileNotFoundError(f"Claude data file not found: {args.claude_data_path}")
    
    # Train models with each reasoning strategy
    for strategy in args.strategies:
        strategy_output_dir = f"{args.output_dir}_{strategy}"
        
        print(f"\n{'='*50}")
        print(f"Starting training with {strategy} reasoning strategy")
        print(f"Output directory: {strategy_output_dir}")
        print(f"{'='*50}\n")
        
        # Build command
        cmd = [
            "python", "-m", "src.main", "sft",
            "--model", "qwen-7b",
            "--claude_data_path", args.claude_data_path,
            "--reasoning_strategy", strategy,
            "--output_dir", strategy_output_dir,
            "--batch_size", str(args.batch_size),
            "--num_epochs", str(args.epochs)
        ]
        
        # Run the command
        try:
            subprocess.run(cmd, check=True)
            print(f"\nTraining with {strategy} strategy completed successfully")
            
            # Find the test set
            test_data_path = f"./data/test/claude_traces_test_set_{strategy}.json"
            if os.path.exists(test_data_path):
                print(f"Test data saved to {test_data_path}")
                print("\nEvaluation options:")
                print("1. Automatic evaluation (recommended):")
                print(f"   ./scripts/evaluate_latest.py")
                print("\n2. REPAIRITY-specific evaluation (for reasoning capabilities):")
                print(f"   python src/evaluation/repairity_eval.py --model_path {strategy_output_dir} --claude_test_path {test_data_path}")
                print("\n3. Standard benchmark evaluation:")
                print(f"   python -m src.main --model qwen-7b --model_path {strategy_output_dir} --output_dir ./eval_results_{strategy}")
                print(f"\nNote: Your trained model is saved at: {strategy_output_dir}")
                print("Remember this path for later evaluation.")
        except subprocess.CalledProcessError as e:
            print(f"\nError in training with {strategy} strategy: {e}")
    
    print("\nAll training runs completed!")

if __name__ == "__main__":
    main() 