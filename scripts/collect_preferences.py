#!/usr/bin/env python3
# collect_preferences.py - Script to collect binary feedback for RL training

import os
import sys
import argparse
import subprocess
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Collect binary feedback from Claude for RL training")
    parser.add_argument("--model_path", required=True, 
                       help="Path to the model to use for generating solutions")
    parser.add_argument("--output_file", default="./data/binary_feedback.json",
                       help="Path to save binary feedback data")
    parser.add_argument("--claude_api_key", default=None,
                       help="Claude API key (otherwise uses environment variable)")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Temperature for generations")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to use")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Set API key in environment if provided
    if args.claude_api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.claude_api_key
        print("Using provided Claude API key")
    elif "ANTHROPIC_API_KEY" in os.environ:
        print("Using Claude API key from environment variable")
    else:
        print("ERROR: No Claude API key provided. Please set ANTHROPIC_API_KEY environment variable or use --claude_api_key")
        sys.exit(1)
    
    print(f"Collecting binary feedback using model: {args.model_path}")
    
    # Import the preference collector from the RL module
    try:
        sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
        from src.rl.preference_collector import collect_preferences
        
        # Run the feedback collection
        collect_preferences({
            "model_path": args.model_path,
            "output_file": args.output_file,
            "temperature": args.temperature,
            "num_samples": args.num_samples
        })
        
        print(f"Binary feedback successfully collected and saved to {args.output_file}")
        
    except ImportError:
        print("Could not import preference collector. Trying alternative methods...")
        
        # Try using commands directly
        try:
            # Try direct import of the module
            cmd = [
                "python", "-m", "src.rl.preference_collector",
                "--model_path", args.model_path,
                "--output_file", args.output_file,
                "--temperature", str(args.temperature),
                "--num_samples", str(args.num_samples)
            ]
            
            if args.claude_api_key:
                cmd.extend(["--claude_api_key", args.claude_api_key])
                
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"Binary feedback successfully collected and saved to {args.output_file}")
                
        except subprocess.CalledProcessError as e:
            print(f"\nFailed to collect binary feedback: {e}")
            print(f"Try running the collector manually: python -m src.rl.preference_collector --model_path {args.model_path} --output_file {args.output_file} --claude_api_key YOUR_API_KEY")
            sys.exit(1)

if __name__ == "__main__":
    main() 