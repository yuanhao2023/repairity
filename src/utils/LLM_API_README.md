# LLM API Setup and Usage

This directory contains utilities for setting up and using various LLM APIs (Claude, OpenAI, etc.) within the TrustCode project.

## Files

- `setup_llm_apis.py` - Utility for setting up API keys and testing connections
- `llm_example.py` - Example script demonstrating how to use the APIs for various tasks
- `claude_client.py` - Basic Claude API client file

## Installation

To use these utilities, you need to install the required packages:

```bash
pip install anthropic python-dotenv
# For other LLMs, you may need:
# pip install openai cohere
```

## API Keys Setup

### Claude API

1. Sign up for an Anthropic API key at [console.anthropic.com](https://console.anthropic.com/)
2. Once you have your API key, you can set it up using:

```bash
# Run the setup or update utility to save your key and test the connection
python src/utils/setup_llm_apis.py claude --api-key "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
" --save --test
```

```bash
# Verify your API key is properly set up
python src/utils/setup_llm_apis.py claude --test
```

This will save your API key to a `.env` file in your home directory and test the connection.

## Usage Examples

### Basic Generation with Claude

```bash
# Generate a general response
python src/utils/llm_example.py --prompt "Explain the concept of recursion in programming"

# Generate code
python src/utils/llm_example.py --mode code --prompt "Write a function to calculate the factorial of a number"

# Generate reasoning trace (for program repair tasks)
python src/utils/llm_example.py --mode reasoning --prompt "Fix this function: def factorial(n): if n <= 0: return 0; return n * factorial(n-1)"
```

### Advanced Usage

```bash
# Choose a specific Claude model
python src/utils/llm_example.py --model claude-3-haiku-20240307 --prompt "Your prompt here"

# Increase retry attempts for busy periods
python src/utils/llm_example.py --retries 5 --prompt "Your prompt here"
```

## Handling API Errors

If you encounter "Overloaded" errors, this means the Claude API is currently experiencing high demand. You can:

1. Try again later
2. Use a less busy model (e.g., claude-3-haiku is often less busy than claude-3-opus or claude-3-sonnet)
3. Reduce the size of your request (lower max_tokens)
4. Increase the retry count and delay

## Integration with the Data Collection Script

The utilities here are designed to work seamlessly with the `claude_data_collector.py` script in the data_processing directory.

## Programmatic Usage

You can use these utilities in your own code:

```python
from src.utils.setup_llm_apis import get_claude_client

# Get a configured Claude client
client = get_claude_client()

# Use the client to generate responses
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1000,
    temperature=0.7,
    messages=[
        {"role": "user", "content": "Your prompt here"}
    ]
)

# Extract the text
text = response.content[0].text
```

## Security Notes

- The API keys are stored in a `.env` file in your home directory
- Never commit API keys to your repository
- Consider using environment variables when deploying in production
- Rotate your API keys periodically for better security 