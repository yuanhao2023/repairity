#!/usr/bin/env python3
# claude_data_collector.py - Collects reasoning traces and solutions from Claude 3.7

import os
import sys
import json
import time
import random
import argparse
from tqdm import tqdm
import anthropic
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import threading
from time import sleep
from concurrent.futures import as_completed

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Load environment variables from .env file
load_dotenv(os.path.expanduser("~/.env"))

# Prompt template for code completion tasks (BigCodeBench)
COMPLETION_PROMPT = """
You are an expert programmer tasked with solving coding problems clearly and efficiently.

Task: {task}

Your output must strictly follow this structure:
1. APPROACH: One paragraph explaining your solution strategy (40-60 words max)
2. CODE: Your complete implementation as a code block

For code completion tasks, ensure your solution seamlessly continues the provided code snippet.
Ensure your code handles all edge cases.
"""

# Original repair-focused prompt - keeping for reference
REASONING_TRACE_PROMPT = """
You are an expert programmer tasked with fixing bugs in code. 
Please analyze the following programming task and provide a solution with a clear reasoning trace.

Task: {task}

Follow these steps:
1. Analyze the problem thoroughly
2. Identify any potential bugs or issues
3. Explain your reasoning for each step
4. Provide the complete corrected solution

Your response should be structured as follows:
- ANALYSIS: Initial analysis of the problem
- ISSUE IDENTIFICATION: Specific bugs or issues found
- REASONING: Step-by-step explanation of your thought process
- SOLUTION: The complete corrected code solution

Make sure your solution is correct, efficient, and follows best practices.
"""

class ClaudeDataCollector:
    def __init__(self, api_key=None, model="claude-3-7-sonnet-20250219", temperature=0.2):
        """Initialize the Claude data collector."""
        # First try the provided API key, then check environment variables
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("No API key provided and ANTHROPIC_API_KEY environment variable not set")
            else:
                print("Using Claude API key from environment variable")
                
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = 5  # Increased from 3 to 5
        self.retry_delay = 10  # Increased from 5 to 10 seconds
        self.requests_per_minute = 30  # Rate limit assumption
        self.last_request_time = 0
        self.call_count = 0
        self.lock = threading.Lock()  # Add lock for thread safety
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API throttling."""
        with self.lock:
            self.call_count += 1
            current_time = time.time()
            # Calculate time since last request
            elapsed = current_time - self.last_request_time
            
            # Determine if we need to sleep to respect rate limits
            # Target: 1 request per (60/requests_per_minute) seconds
            target_delay = 60.0 / self.requests_per_minute
            if elapsed < target_delay:
                sleep_time = target_delay - elapsed
                time.sleep(sleep_time)
                
            # Update last request time
            self.last_request_time = time.time()
    
    def get_reasoning_trace(self, example):
        """Query Claude to get a solution with reasoning for a given example."""
        # Ensure example is a dictionary
        if isinstance(example, str):
            try:
                example = json.loads(example)
            except json.JSONDecodeError:
                # If it's just a string, treat it as the task itself
                task_id = "unknown"
                task = example
                return {
                    "task_id": task_id,
                    "task": task,
                    "reasoning_trace": None,
                    "error": "Example was a string, not a dictionary",
                    "example": {"task": task}
                }
        
        # If we still don't have a dict, try to convert or create a minimal valid dict
        if not isinstance(example, dict):
            try:
                example = dict(example)
            except (TypeError, ValueError):
                print(f"Warning: Example is not a dictionary or convertible to one: {type(example)}")
                task_id = "unknown"
                task = str(example)
                return {
                    "task_id": task_id,
                    "task": task,
                    "reasoning_trace": None,
                    "error": f"Example is not a dictionary: {type(example)}",
                    "example": {"task": task}
                }
        
        # Extract task information
        task_id = example.get("task_id", "")
        task_description = example.get("description", "")
        instruct_prompt = example.get("instruct_prompt", "")
        code_prompt = example.get("code_prompt", "")
        complete_prompt = example.get("complete_prompt", "")
        
        # Construct the task prompt
        if instruct_prompt:
            task = instruct_prompt
        elif complete_prompt:
            task = complete_prompt
        elif code_prompt:
            task = f"Complete the following code:\n\n{code_prompt}"
        else:
            task = f"Task ID: {task_id}\n{task_description}" if task_description else f"Solve task with ID: {task_id}"
        
        # Format the prompt for BigCodeBench
        formatted_prompt = COMPLETION_PROMPT.format(task=task)
        
        # Add special handling for code completion tasks
        if "Complete the following code" in task:
            formatted_prompt += "\nImportant: This is a code completion task. Continue the given code snippet without rewriting it."
        
        # Set system message for quality code generation
        system_message = "You are an expert programmer. Generate clean, efficient, and correct code that follows best practices."
        
        # Apply rate limiting before making the request
        self._rate_limit()
        
        # Query Claude with retries and exponential backoff
        errors = []
        for attempt in range(self.max_retries):
            try:
                # Adjust max tokens based on attempt to reduce load on later attempts
                max_tokens = 2500 - (attempt * 200)  # Gradually decrease token count on retries
                max_tokens = max(1500, max_tokens)  # But don't go below 1500
                
                # Use messages API with system message
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    system=system_message,
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ]
                )
                
                # Check if response seems complete (contains code block)
                content = response.content[0].text
                if "```" not in content:
                    errors.append(f"Response missing code block (attempt {attempt+1})")
                    # If this is the last attempt, return whatever we got
                    if attempt == self.max_retries - 1:
                        print(f"Warning: Response for task {task_id} missing code block after all retries")
                        break
                    # Otherwise retry with a clearer prompt
                    formatted_prompt += "\nMake sure to include your code in a code block between ``` markers."
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                    
                return {
                    "task_id": task_id,
                    "task": task,
                    "reasoning_trace": content,
                    "model": self.model,
                    "example": example
                }
            except Exception as e:
                error_str = str(e)
                errors.append(f"Attempt {attempt+1}: {error_str}")
                
                # Determine if we should retry
                should_retry = attempt < self.max_retries - 1
                wait_time = 0
                
                # Handle different error types
                if "overloaded" in error_str.lower() and should_retry:
                    # For overloaded errors, use longer exponential backoff
                    wait_time = self.retry_delay * (2 ** (attempt + 1))
                    # If we've hit overloaded errors twice in a row, reduce load factors
                    if attempt >= 1:
                        print(f"API overloaded, reducing request complexity")
                        max_tokens = 1000  # Drastically reduce tokens
                    print(f"API overloaded. Retrying in {wait_time} seconds...")
                elif "rate_limited" in error_str.lower() and should_retry:
                    # For rate limit errors, use longer backoff and reduce our rate
                    wait_time = self.retry_delay * (3 ** (attempt + 1))
                    self.requests_per_minute = max(1, self.requests_per_minute // 2)  # Cut rate limit in half
                    print(f"Rate limited. Reducing to {self.requests_per_minute} RPM. Retrying in {wait_time} seconds...")
                elif should_retry:
                    # For other errors, use standard exponential backoff
                    wait_time = self.retry_delay * (1.5 ** attempt)
                    print(f"Error: {error_str}. Retrying in {wait_time} seconds...")
                else:
                    print(f"Failed after {self.max_retries} attempts: {error_str}")
                    return {
                        "task_id": task_id,
                        "task": task,
                        "reasoning_trace": None,
                        "error": error_str,
                        "all_errors": errors,
                        "example": example
                    }
                
                if wait_time > 0:
                    time.sleep(wait_time)
        
        # If we exhausted all retries with no success
        return {
            "task_id": task_id,
            "task": task,
            "reasoning_trace": None,
            "error": "Max retries exceeded",
            "all_errors": errors,
            "example": example
        }
    
    def extract_sections(self, reasoning_trace):
        """Extract the different sections from a reasoning trace."""
        if not reasoning_trace:
            return None
        
        # Try to find sections in the reasoning trace
        sections = {}
        current_section = "preamble"
        current_content = []
        
        for line in reasoning_trace.split("\n"):
            # Check for section headers - support both old and new formats
            if line.strip().upper().startswith("BRIEF_THOUGHT_PROCESS:") or "BRIEF_THOUGHT_PROCESS" == line.strip().upper():
                sections["preamble"] = "\n".join(current_content).strip()
                current_section = "brief_thought_process"
                current_content = []
                continue
            elif line.strip().upper().startswith("ANALYSIS:") or "ANALYSIS" == line.strip().upper():
                sections["preamble"] = "\n".join(current_content).strip()
                current_section = "analysis"
                current_content = []
                continue
            elif line.strip().upper().startswith("ISSUE") or "ISSUE IDENTIFICATION" == line.strip().upper():
                sections[current_section] = "\n".join(current_content).strip()
                current_section = "issue_identification"
                current_content = []
                continue
            elif line.strip().upper().startswith("REASONING:") or "REASONING" == line.strip().upper():
                sections[current_section] = "\n".join(current_content).strip()
                current_section = "reasoning"
                current_content = []
                continue
            elif line.strip().upper().startswith("SOLUTION:") or "SOLUTION" == line.strip().upper():
                sections[current_section] = "\n".join(current_content).strip()
                current_section = "solution"
                current_content = []
                continue
            
            # Add line to current section
            current_content.append(line)
        
        # Add the last section
        sections[current_section] = "\n".join(current_content).strip()
        
        # If no sections were found, try a different approach
        if len(sections) <= 2:  # Just preamble and possibly one other section
            # Look for code blocks
            code_blocks = []
            in_code_block = False
            code_block = []
            
            for line in reasoning_trace.split("\n"):
                if line.strip().startswith("```"):
                    if in_code_block:
                        code_blocks.append("\n".join(code_block))
                        code_block = []
                    in_code_block = not in_code_block
                    continue
                
                if in_code_block:
                    code_block.append(line)
            
            # If code blocks were found, use the last one as the solution
            if code_blocks:
                # For code completion tasks, add the reasoning as brief_thought_process
                if "BRIEF_THOUGHT_PROCESS" in reasoning_trace:
                    solution = code_blocks[-1]
                    reasoning = reasoning_trace.split("```")[0].strip()  # Everything before the first code block
                    sections = {
                        "brief_thought_process": reasoning,
                        "solution": solution
                    }
                else:
                    solution = code_blocks[-1]
                    reasoning = reasoning_trace.split("```")[0].strip()  # Everything before the first code block
                    sections = {
                        "reasoning": reasoning,
                        "solution": solution
                    }
        
        return sections
    
    def extract_solution(self, reasoning_trace):
        """Extract just the solution code from a reasoning trace."""
        if not reasoning_trace:
            return None
        
        sections = self.extract_sections(reasoning_trace)
        if not sections:
            return None
        
        # Try to get the solution from the sections
        if "solution" in sections:
            solution = sections["solution"]
            
            # Check if the solution is wrapped in a code block
            if "```" in solution:
                # Extract from code block
                parts = solution.split("```")
                for i, part in enumerate(parts):
                    if i > 0 and i < len(parts) - 1 and not part.strip().startswith(("python", "java", "c++", "javascript")):
                        return part.strip()
                # If we didn't find a code block in the middle, try the last part
                if len(parts) > 1:
                    return parts[1].strip()
            
            return solution
        
        # Fallback: look for code blocks in the entire trace
        code_blocks = []
        in_code_block = False
        code_block = []
        
        for line in reasoning_trace.split("\n"):
            if line.strip().startswith("```"):
                if in_code_block:
                    code_blocks.append("\n".join(code_block))
                    code_block = []
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                code_block.append(line)
        
        # Use the last code block as the solution
        if code_blocks:
            return code_blocks[-1]
        
        return None

def collect_data(args):
    """Main function to collect data from Claude 3.7."""
    try:
        # Print API key status (for debugging)
        if args.api_key:
            print(f"Using API key provided via command line argument")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            print(f"Using API key from environment variable (length: {len(os.environ.get('ANTHROPIC_API_KEY'))})")
        else:
            print("No API key found in arguments or environment variables")
            return
        
        # Initialize collector with custom retry settings if provided
        collector = ClaudeDataCollector(
            api_key=args.api_key, 
            model=args.model, 
            temperature=args.temperature
        )
        
        # Override retry settings if provided
        if hasattr(args, 'retry_delay') and args.retry_delay:
            collector.retry_delay = args.retry_delay
            print(f"Using custom retry delay: {collector.retry_delay} seconds")
            
        if hasattr(args, 'max_retries') and args.max_retries:
            collector.max_retries = args.max_retries
            print(f"Using custom max retries: {collector.max_retries}")
            
        print(f"Successfully initialized Claude client with model: {args.model}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Check for existing output file to resume data collection
        output_file = os.path.join(args.output_dir, f"claude_reasoning_traces_{args.version}_{args.num_examples}.json")
        
        existing_data = []
        existing_task_ids = set()
        
        if os.path.exists(output_file) and args.resume:
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                existing_task_ids = {item["task_id"] for item in existing_data}
                print(f"Found {len(existing_data)} existing examples, will resume collection")
            except Exception as e:
                print(f"Error loading existing data, starting fresh: {e}")
                existing_data = []
        
        # Load the dataset
        print(f"Loading BigCodeBench dataset...")
        dataset = load_dataset("bigcode/bigcodebench")
        
        if args.version not in dataset:
            print(f"Version {args.version} not found in dataset. Available versions: {list(dataset.keys())}")
            return
        
        # Get the dataset for the specified version
        version_data = dataset[args.version]
        
        # Print dataset information for debugging
        print(f"Dataset type: {type(version_data)}")
        print(f"Dataset size: {len(version_data)} examples")
        print(f"Dataset features: {version_data.features}")
        if len(version_data) > 0:
            print(f"First example keys: {list(version_data[0].keys()) if hasattr(version_data[0], 'keys') else 'No keys method'}")
            print(f"Example accessing first item: task_id = {version_data[0]['task_id']}")
        
        # Determine examples to process
        if args.num_examples > 0 and args.num_examples < len(version_data):
            if args.random_sample:
                # Set seed for reproducibility
                random.seed(args.seed)
                all_indices = list(range(len(version_data)))
                indices = random.sample(all_indices, args.num_examples)
            else:
                indices = list(range(args.num_examples))
            examples = version_data.select(indices)
        else:
            examples = version_data
            args.num_examples = len(examples)
            
        # Debug: Print example format information
        if len(examples) > 0:
            sample_example = examples[0]
            print(f"Dataset example type: {type(sample_example)}")
            print(f"Dataset example fields: {list(sample_example.keys()) if hasattr(sample_example, 'keys') else 'No keys method'}")
            print(f"Converting dataset items to dictionaries for processing...")
        
        # Filter out examples that have already been processed
        if existing_task_ids and args.resume:
            examples_to_process = []
            for example in examples:
                if example["task_id"] not in existing_task_ids:
                    examples_to_process.append(example)
            
            print(f"Processing {len(examples_to_process)} new examples out of {len(examples)} total")
            
            # If all examples have been processed, we're done
            if not examples_to_process:
                print("All examples have already been processed. No new data to collect.")
                return
            
            examples = examples_to_process
        else:
            print(f"Processing {len(examples)} examples from BigCodeBench {args.version}...")
        
        # Process in batches to allow for regular saving
        batch_size = min(args.batch_size, len(examples))
        num_batches = (len(examples) + batch_size - 1) // batch_size
        
        all_results = existing_data.copy() if args.resume else []
        successful = len([x for x in all_results if x.get("solution")])
        
        # Print dataset iteration method
        print("Converting dataset to list for processing...")
        # Convert dataset to a list of dictionaries for easier handling
        examples_list = []
        for i in range(len(examples)):
            examples_list.append(examples[i])
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(examples_list))
            batch_examples = examples_list[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_idx+1}/{num_batches} ({batch_start+1}-{batch_end} of {len(examples_list)})")
            
            # Collect data with progress bar for this batch
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = []
                for example in batch_examples:
                    futures.append(executor.submit(collector.get_reasoning_trace, example))
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_idx+1}"):
                    result = future.result()
                    batch_results.append(result)
            
            # Process batch results
            batch_successful = 0
            processed_batch = []
            
            for result in batch_results:
                if result["reasoning_trace"]:
                    # Extract solution
                    solution = collector.extract_solution(result["reasoning_trace"])
                    
                    # Only count as successful if we got a solution
                    if solution:
                        batch_successful += 1
                        successful += 1
                    
                    # Extract sections
                    sections = collector.extract_sections(result["reasoning_trace"])
                    
                    processed_result = {
                        "task_id": result["task_id"],
                        "task": result["task"],
                        "reasoning_trace": result["reasoning_trace"],
                        "solution": solution,
                        "sections": sections,
                        "model": result["model"],
                        "reference_solution": result["example"].get("canonical_solution", "")
                    }
                    processed_batch.append(processed_result)
                else:
                    print(f"Failed to get response for task {result['task_id']}: {result.get('error', 'Unknown error')}")
            
            # Add batch results to overall results
            all_results.extend(processed_batch)
            
            # Save intermediate results after each batch
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)
            
            print(f"Batch {batch_idx+1} results: {batch_successful}/{len(batch_examples)} successful")
            print(f"Cumulative results: {successful}/{len(all_results)} successful")
            print(f"Saved intermediate results to {output_file}")
            
            # If we've collected enough successful examples, we can stop
            if args.stop_at_success and successful >= args.stop_at_success:
                print(f"Reached target of {args.stop_at_success} successful examples. Stopping collection.")
                break
        
        print(f"\nData collection complete. Final results: {successful}/{len(all_results)} successful examples")
        print(f"Results saved to {output_file}")
    
    except Exception as e:
        print(f"Error in data collection: {e}")
        print("If this is an authentication error, please check your API key setup.")
        import traceback
        traceback.print_exc()
        print("You can set up your API key using: python src/utils/setup_llm_apis.py claude --api-key YOUR_API_KEY --save")

def main():
    parser = argparse.ArgumentParser(description="Collect reasoning traces from Claude 3.7 for BigCodeBench")
    parser.add_argument("--api_key", default=None, help="Anthropic API key (default: use environment variable)")
    parser.add_argument("--model", default="claude-3-7-sonnet-20250219", help="Claude model to use")
    parser.add_argument("--version", default="v0.1.4", help="BigCodeBench version")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to process")
    parser.add_argument("--random_sample", action="store_true", help="Randomly sample examples instead of taking first N")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for Claude generation")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent workers")
    parser.add_argument("--output_dir", default="./data", help="Directory to save results")
    parser.add_argument("--resume", action="store_true", help="Resume data collection from existing file")
    parser.add_argument("--stop_at_success", type=int, help="Stop collection after reaching this many successful examples")
    parser.add_argument("--seed", type=int, help="Seed for random sampling")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()
    collect_data(args)

if __name__ == "__main__":
    main() 