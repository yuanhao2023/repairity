#!/usr/bin/env python3
# prepare_bigcodebench_data.py - Convert BigCodeBench data to format for RLLF training

import os
import json
import argparse
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert BigCodeBench data for RLLF training")
    parser.add_argument(
        "--input_files", 
        nargs="+",
        required=True,
        help="Path to input files (binary_feedback.json and test files)"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="./data/bigcode-evaluation-data.jsonl",
        help="Path to output JSONL file for RLLF training"
    )
    return parser.parse_args()

def convert_binary_feedback(input_path):
    """Convert binary feedback JSON to RLLF format."""
    logger.info(f"Processing binary feedback from {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    converted_data = []
    for item in data:
        # Extract just the problem statement for training
        converted_item = {
            "id": item.get("sample_id", ""),
            "input": item.get("input", ""),
            "label": item.get("label", "")  # Keep the GOOD/BAD label for reference
        }
        converted_data.append(converted_item)
    
    return converted_data

def convert_test_data(input_path):
    """Convert test set JSON to RLLF format."""
    logger.info(f"Processing test data from {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    converted_data = []
    for item in data:
        # Extract just the problem statement for training
        converted_item = {
            "id": item.get("task_id", ""),
            "input": item.get("task", "")
        }
        converted_data.append(converted_item)
    
    return converted_data

def main():
    args = parse_args()
    
    all_data = []
    
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            logger.error(f"File not found: {input_file}")
            continue
        
        if "binary_feedback" in input_file:
            data = convert_binary_feedback(input_file)
        else:
            data = convert_test_data(input_file)
        
        all_data.extend(data)
    
    logger.info(f"Converted {len(all_data)} items total")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Write to JSONL format
    with open(args.output_file, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Output written to {args.output_file}")

if __name__ == "__main__":
    main() 