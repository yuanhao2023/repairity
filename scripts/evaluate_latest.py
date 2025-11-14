#!/usr/bin/env python3
# evaluate_latest.py - Script to find and evaluate the latest repairity model

import os
import glob
import argparse
import subprocess
from datetime import datetime

def find_latest_model(base_dir="./models", model_prefix="qwen_repairity_", strategy=None):
    """Find the latest qwen_repairity model."""
    # Create the pattern to search for
    if strategy:
        pattern = os.path.join(base_dir, f"{model_prefix}*_{strategy}")
    else:
        pattern = os.path.join(base_dir, f"{model_prefix}*")
    
    # Find all matching directories
    matching_dirs = glob.glob(pattern)
    
    if not matching_dirs:
        if strategy:
            raise FileNotFoundError(f"No models found with pattern {pattern}")
        else:
            raise FileNotFoundError(f"No models found in {base_dir} with prefix {model_prefix}")
    
    # Sort by modification time (newest last)
    latest_dir = max(matching_dirs, key=os.path.getmtime)
    
    # Extract the strategy if not provided
    if not strategy:
        # The strategy is the last part after the last underscore
        strategy = latest_dir.split("_")[-1]
    
    return latest_dir, strategy

def main():
    parser = argparse.ArgumentParser(description="Evaluate the latest qwen_repairity model")
    parser.add_argument("--strategy", choices=["explicit", "cot", "alternating", "none"], 
                        help="Strategy to evaluate (if not provided, the latest model of any strategy will be used)")
    parser.add_argument("--eval_type", choices=["standard", "repairity", "both"], default="both",
                        help="Type of evaluation to run")
    parser.add_argument("--model_dir", default="./models", 
                        help="Directory containing model checkpoints")
    
    args = parser.parse_args()
    
    try:
        # Find the latest model
        model_path, strategy = find_latest_model(
            base_dir=args.model_dir, 
            strategy=args.strategy
        )
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_output_dir = f"./eval_results_{strategy}_{timestamp}"
        
        # Create the output directory if it doesn't exist
        os.makedirs(eval_output_dir, exist_ok=True)
        
        print(f"Found latest model: {model_path}")
        print(f"Strategy detected: {strategy}")
        print(f"Evaluation results will be saved to: {eval_output_dir}")
        
        # Run evaluation(s)
        if args.eval_type in ["standard", "both"]:
            print("\nRunning standard benchmark evaluation...")
            standard_cmd = [
                "python", "scripts/run_eval.py",
                "--model_path", model_path,
                "--output_dir", eval_output_dir
            ]
            print(f"Debug - Running command: {' '.join(standard_cmd)}")
            
            # Run in shell mode to ensure proper argument handling
            subprocess.run(" ".join(standard_cmd), shell=True, check=True)
        
        if args.eval_type in ["repairity", "both"]:
            print("\nRunning REPAIRITY-specific evaluation...")
            # Check if test data exists
            test_data_path = f"./data/test/claude_traces_test_set_{strategy}.json"
            if os.path.exists(test_data_path):
                repairity_cmd = [
                    "python", "src/evaluation/repairity_eval.py",
                    "--model_path", model_path,
                    "--claude_test_path", test_data_path,
                    "--output_file", f"{eval_output_dir}/repairity_eval_results.json"
                ]
                print(f"Debug - Running command: {' '.join(repairity_cmd)}")
                
                # Run in shell mode to ensure proper argument handling
                subprocess.run(" ".join(repairity_cmd), shell=True, check=True)
            else:
                print(f"Warning: Could not find test data at {test_data_path}")
                print("Skipping REPAIRITY-specific evaluation")
        
        print(f"\nEvaluation complete. Results saved to {eval_output_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that you have trained models in the specified directory.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main() 