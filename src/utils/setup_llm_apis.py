#!/usr/bin/env python3
# setup_llm_apis.py - Utility for setting up access to LLM APIs

import os
import argparse
import sys
import logging
from dotenv import load_dotenv

# Try importing LLM libraries
# We'll catch import errors and provide installation instructions
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def check_api_dependencies():
    """Check if required packages are installed and provide installation instructions if needed."""
    missing_deps = []
    
    if not ANTHROPIC_AVAILABLE:
        missing_deps.append("anthropic")
    
    if not OPENAI_AVAILABLE:
        missing_deps.append("openai")
    
    if not COHERE_AVAILABLE:
        missing_deps.append("cohere")
    
    if not os.path.exists(os.path.expanduser("~/.env")):
        logger.warning("No .env file found in home directory. Creating one for API keys.")
        with open(os.path.expanduser("~/.env"), "w") as f:
            f.write("# LLM API Keys - KEEP THIS FILE SECURE\n")
            f.write("# Claude (Anthropic) API Key\n")
            f.write("ANTHROPIC_API_KEY=\n\n")
            f.write("# OpenAI API Key\n")
            f.write("OPENAI_API_KEY=\n\n")
            f.write("# Cohere API Key\n")
            f.write("COHERE_API_KEY=\n")
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
        install_cmd = f"pip install {' '.join(missing_deps)}"
        logger.info(f"Install with: {install_cmd}")
        return False
    
    return True

def test_claude_api(api_key=None):
    """Test Claude API connection."""
    if not ANTHROPIC_AVAILABLE:
        logger.error("Claude API client not installed. Run 'pip install anthropic'")
        return False
    
    try:
        # First try to use provided key, then env var, then check .env file
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                load_dotenv(os.path.expanduser("~/.env"))
                api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            logger.error("No Claude API key found. Please provide a key.")
            return False
        
        # Initialize Claude client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Test with a simple query
        message = client.messages.create(
            model="claude-3-opus-20240229",  # Use appropriate model
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": "Say hello"}]
        )
        
        logger.info(f"Claude API test successful. Response: {message.content[0].text}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing Claude API: {str(e)}")
        return False

def test_openai_api(api_key=None):
    """Test OpenAI API connection."""
    if not OPENAI_AVAILABLE:
        logger.error("OpenAI API client not installed. Run 'pip install openai'")
        return False
    
    try:
        # Use provided key or check environment variables
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                load_dotenv(os.path.expanduser("~/.env"))
                api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            logger.error("No OpenAI API key found. Please provide a key.")
            return False
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple query
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        logger.info(f"OpenAI API test successful. Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing OpenAI API: {str(e)}")
        return False

def save_api_key(service, api_key):
    """Save API key to .env file."""
    env_file = os.path.expanduser("~/.env")
    env_var = f"{service.upper()}_API_KEY"
    
    if os.path.exists(env_file):
        # Read current content
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Check if key already exists
        key_exists = False
        for i, line in enumerate(lines):
            if line.startswith(f"{env_var}="):
                lines[i] = f"{env_var}={api_key}\n"
                key_exists = True
                break
        
        # Add key if it doesn't exist
        if not key_exists:
            lines.append(f"{env_var}={api_key}\n")
        
        # Write back
        with open(env_file, 'w') as f:
            f.writelines(lines)
    else:
        # Create new file
        with open(env_file, 'w') as f:
            f.write(f"{env_var}={api_key}\n")
    
    logger.info(f"Saved {service} API key to {env_file}")
    # Set environment variable for current session
    os.environ[env_var] = api_key

def get_claude_client(api_key=None):
    """Get a configured Claude client."""
    if not ANTHROPIC_AVAILABLE:
        logger.error("Claude API client not installed. Run 'pip install anthropic'")
        return None
    
    # Use provided key or check environment variables
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            load_dotenv(os.path.expanduser("~/.env"))
            api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        logger.error("No Claude API key found. Please provide a key.")
        return None
    
    return anthropic.Anthropic(api_key=api_key)

def main():
    parser = argparse.ArgumentParser(description="Set up access to various LLM APIs")
    
    # Common options
    parser.add_argument("--check-deps", action="store_true", 
                        help="Check if all API client dependencies are installed")
    
    # LLM-specific options
    subparsers = parser.add_subparsers(dest="llm", help="LLM service to configure")
    
    # Claude/Anthropic options
    claude_parser = subparsers.add_parser("claude", help="Configure Claude API")
    claude_parser.add_argument("--api-key", help="Claude API key")
    claude_parser.add_argument("--test", action="store_true", help="Test Claude API connection")
    claude_parser.add_argument("--save", action="store_true", help="Save API key to .env file")
    
    # OpenAI options
    openai_parser = subparsers.add_parser("openai", help="Configure OpenAI API")
    openai_parser.add_argument("--api-key", help="OpenAI API key")
    openai_parser.add_argument("--test", action="store_true", help="Test OpenAI API connection")
    openai_parser.add_argument("--save", action="store_true", help="Save API key to .env file")
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        check_api_dependencies()
        return
    
    # Handle Claude setup
    if args.llm == "claude":
        if args.save and args.api_key:
            save_api_key("anthropic", args.api_key)
        
        if args.test:
            test_claude_api(args.api_key)
    
    # Handle OpenAI setup
    elif args.llm == "openai":
        if args.save and args.api_key:
            save_api_key("openai", args.api_key)
        
        if args.test:
            test_openai_api(args.api_key)
    
    # Default behavior - provide info
    else:
        parser.print_help()
        
        print("\nExample usage:")
        print("  Check dependencies:")
        print("    python setup_llm_apis.py --check-deps")
        print("\n  Configure Claude API:")
        print("    python setup_llm_apis.py claude --api-key YOUR_API_KEY --save --test")
        print("\n  Configure OpenAI API:")
        print("    python setup_llm_apis.py openai --api-key YOUR_API_KEY --save --test")

if __name__ == "__main__":
    main() 