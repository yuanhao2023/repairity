#!/usr/bin/env python3
# main.py - Main entry point for TrustCode project

import argparse
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def main():
    parser = argparse.ArgumentParser(description="TrustCode - Enhancing LLMs for Program Repair")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # SFT training commands
    sft_parser = subparsers.add_parser("sft", help="Supervised Fine-Tuning")
    sft_parser.add_argument("--model", choices=["qwen-7b", "swebench"], required=True, 
                            help="Model to fine-tune")
    sft_parser.add_argument("--output_dir", default=None, help="Output directory for the fine-tuned model")
    sft_parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    sft_parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    sft_parser.add_argument("--claude_data_path", default=None, 
                            help="Path to Claude reasoning traces JSON file (if using Claude data)")
    sft_parser.add_argument("--reasoning_strategy", choices=["explicit", "cot", "alternating", "none"], 
                            default="explicit", help="Strategy to format reasoning traces")
    
    # Evaluation commands
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate fine-tuned models")
    eval_parser.add_argument("--model", choices=["qwen-7b", "swebench"], required=True,
                          help="Model to evaluate")
    eval_parser.add_argument("--model_path", required=True, help="Path to the fine-tuned model")
    eval_parser.add_argument("--output_dir", default="./eval_results", help="Directory to save evaluation results")
    
    # Data collection commands
    data_parser = subparsers.add_parser("collect-data", help="Collect data from Claude 3.7")
    data_parser.add_argument("--api_key", default=None, help="Anthropic API key (default: use environment variable)")
    data_parser.add_argument("--model", default="claude-3-7-sonnet-20250219", help="Claude model to use")
    data_parser.add_argument("--version", default="v0.1.4", help="BigCodeBench version")
    data_parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to process")
    data_parser.add_argument("--random_sample", action="store_true", help="Randomly sample examples")
    data_parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for Claude generation")
    data_parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers")
    data_parser.add_argument("--output_dir", default="./data", help="Directory to save results")
    data_parser.add_argument("--retry_delay", type=int, default=10, help="Base delay between retries in seconds")
    data_parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retry attempts")
    data_parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    data_parser.add_argument("--resume", action="store_true", help="Resume data collection from existing file")
    data_parser.add_argument("--stop_at_success", type=int, help="Stop collection after reaching this many successful examples")
    data_parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling")
    
    # RL-LF commands
    rl_parser = subparsers.add_parser("rl", help="Reinforcement Learning with LLM Feedback")
    rl_subparsers = rl_parser.add_subparsers(dest="rl_command", help="RL command to execute")
    
    # Collect preferences
    pref_parser = rl_subparsers.add_parser("collect-preferences", help="Collect preferences from LLM")
    pref_parser.add_argument("--model_path", required=True, help="Path to the SFT model")
    pref_parser.add_argument("--output_file", required=True, help="File to save collected preferences")
    
    # Train reward model
    reward_parser = rl_subparsers.add_parser("train-reward", help="Train reward model")
    reward_parser.add_argument("--preferences_file", required=True, help="File with collected preferences")
    reward_parser.add_argument("--output_dir", required=True, help="Directory to save reward model")
    
    # PPO training
    ppo_parser = rl_subparsers.add_parser("ppo", help="PPO fine-tuning")
    ppo_parser.add_argument("--sft_model_path", required=True, help="Path to the SFT model")
    ppo_parser.add_argument("--reward_model_path", required=True, help="Path to the reward model")
    ppo_parser.add_argument("--output_dir", required=True, help="Directory to save PPO-fine-tuned model")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "sft":
        if args.model == "qwen-7b":
            from src.training.qwen_sft import fine_tune
            fine_tune(args)
        elif args.model == "swebench":
            from src.training.swebench_sft import fine_tune
            fine_tune(args)
    
    elif args.command == "evaluate":
        if args.model == "qwen-7b":
            from src.evaluation.qwen_eval import evaluate
            evaluate(args)
        elif args.model == "swebench":
            from src.evaluation.swebench_eval import evaluate
            evaluate(args)
    
    elif args.command == "collect-data":
        from src.data_processing.claude_data_collector import collect_data
        collect_data(args)
    
    elif args.command == "rl":
        if args.rl_command == "collect-preferences":
            from src.rl.preference_collector import collect_preferences
            collect_preferences(args)
        elif args.rl_command == "train-reward":
            from src.rl.train_reward_model import train_reward_model
            train_reward_model(args)
        elif args.rl_command == "ppo":
            from src.rl.ppo_training import train_ppo
            train_ppo(args)
        else:
            rl_parser.print_help()
    
if __name__ == "__main__":
    main() 