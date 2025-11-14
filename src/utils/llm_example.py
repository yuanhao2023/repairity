#!/usr/bin/env python3
# llm_example.py - Example of using LLM APIs in the project

import os
import sys
import time
import argparse
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.utils.setup_llm_apis import get_claude_client

def generate_with_claude(prompt, model="claude-3-7-sonnet-20250219", max_tokens=1000, 
                         temperature=0.7, retries=3, retry_delay=2):
    """
    Generate a response using Claude.
    
    Args:
        prompt: The prompt to send to Claude
        model: The Claude model to use
        max_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (0-1)
        retries: Number of retries if the API is overloaded
        retry_delay: Base delay between retries (will be exponentially increased)
    
    Returns:
        Generated text or None if an error occurred
    """
    # Load API key from environment variables or .env file
    load_dotenv()
    
    # Get Claude client
    client = get_claude_client()
    if not client:
        print("Failed to initialize Claude client. Please check your API key.")
        return None
    
    for attempt in range(retries):
        try:
            # Create a message
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Return the response text
            return message.content[0].text
            
        except Exception as e:
            error_str = str(e)
            if "overloaded" in error_str.lower() and attempt < retries - 1:
                current_delay = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"API overloaded. Retrying in {current_delay} seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(current_delay)
            else:
                print(f"Error generating with Claude: {error_str}")
                if "overloaded" in error_str.lower():
                    print("\nThe Claude API is currently experiencing high demand. Please try again later or:")
                    print("1. Try a different model like 'claude-3-haiku-20240307' which may have more capacity")
                    print("2. Reduce the max_tokens requested")
                    print("3. Try during non-peak hours")
                return None
    
    return None

def generate_code_with_claude(prompt, language="python", model="claude-3-7-sonnet-20250219"):
    """Generate code using Claude with a specific programming task."""
    # Create a code-specific prompt
    code_prompt = f"""
You are an expert {language} programmer. Please write code to solve the following task:

{prompt}

Provide only the code without explanations.
Make sure your solution is correct, efficient, and follows best practices.
"""
    
    # Generate the response
    response = generate_with_claude(code_prompt, model=model, temperature=0.3)
    return response

def generate_reasoning_trace(prompt, model="claude-3-7-sonnet-20250219"):
    """Generate a reasoning trace for a programming problem."""
    reasoning_prompt = f"""
You are an expert programmer tasked with fixing bugs in code. 
Please analyze the following programming task and provide a solution with a clear reasoning trace.

Task: {prompt}

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
    
    # Generate the response
    response = generate_with_claude(reasoning_prompt, model=model, temperature=0.1, max_tokens=4000)
    return response

def main():
    parser = argparse.ArgumentParser(description="Example of using LLM APIs")
    parser.add_argument("--prompt", default="Write a function to calculate the Fibonacci sequence.", 
                        help="Prompt to send to the LLM")
    parser.add_argument("--model", default="claude-3-7-sonnet-20250219",
                        help="Claude model to use")
    parser.add_argument("--mode", choices=["general", "code", "reasoning"], default="general",
                        help="Generation mode")
    parser.add_argument("--retries", type=int, default=3,
                        help="Number of retries if the API is overloaded")
    
    args = parser.parse_args()
    
    print(f"Using Claude model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Sending prompt: \"{args.prompt}\"")
    print("Generating response...")
    
    if args.mode == "general":
        response = generate_with_claude(args.prompt, model=args.model, retries=args.retries)
    elif args.mode == "code":
        response = generate_code_with_claude(args.prompt, model=args.model)
    elif args.mode == "reasoning":
        response = generate_reasoning_trace(args.prompt, model=args.model)
    
    if response:
        print("\n" + "="*80)
        print(f"Response from {args.model} ({args.mode} mode):")
        print("="*80)
        print(response)
        print("="*80)
    else:
        print("\nFailed to get a response. Please check the error messages above.")

if __name__ == "__main__":
    main() 