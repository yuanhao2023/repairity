# REPAIRITY Training Guide

This document explains how to implement the REPAIRITY approach for enhancing open-source LLMs with slow-thinking abilities.

## REPAIRITY Methodology

REPAIRITY consists of three primary steps:

1. **Data Collection from Strong LLMs**: Collecting reasoning traces and solutions from closed-source LLMs (Claude 3.7 in our case)
2. **Reasoning Trace Learning**: Fine-tuning an open-source LLM (Qwen 7B) to mimic the reasoning traces
3. **Reinforcement Learning with LLM Feedback**: Further refining the model with RL based on feedback

This guide focuses on the first two steps.

## Step 1: Data Collection

We collect reasoning traces from Claude 3.7 using the `claude_data_collector.py` script:

```bash
python -m src.main collect-data --num_examples 100 --batch_size 10 --output_dir ./data
```

This will create a JSON file like `claude_reasoning_traces_v0.1.4_100.json` containing:
- Task descriptions from BigCodeBench
- Claude's reasoning traces with structured approach explanations
- Final code solutions

## Step 2: Reasoning Trace Learning

### Data Splitting

The collected data is automatically split into training and test sets (default: 90% training, 10% testing). The test set is saved separately for proper evaluation.

### Reasoning Strategies

We've implemented multiple strategies for the slow-thinking training:

1. **Explicit Reasoning** (`explicit`): Clear separation between reasoning and code
   ```
   ## APPROACH
   [explanation of the approach]
   
   ## CODE
   ```python
   [solution code]
   ```
   ```

2. **Chain-of-Thought** (`cot`): Step-by-step reasoning process
   ```
   First, I'll analyze the problem: [analysis]
   
   Key considerations: [issues identified]
   
   My approach: [reasoning]
   
   Final solution:
   ```python
   [solution code]
   ```
   ```

3. **Alternating Reasoning** (`alternating`): Interleaving explanation with code segments
   ```
   First, I'll consider my overall approach: [approach]
   
   Part 1 of the solution:
   ```python
   [code part 1]
   ```
   
   Part 2 of the solution:
   ```python
   [code part 2]
   ```
   ```

4. **No Reasoning** (`none`): Just the solution code (baseline)

### Training Command

To train Qwen with different reasoning strategies:

```bash
./train_qwen_with_claude.py --strategies explicit cot alternating --epochs 3 --batch_size 4
```

Or run a specific strategy:

```bash
python -m src.main sft --model qwen-7b --claude_data_path ./data/claude_reasoning_traces_v0.1.4_100.json --reasoning_strategy explicit
```

### Expected Results

After training, the model should be able to:
1. Generate high-quality code with structured reasoning
2. Explain its approach before providing a solution
3. Break down problem-solving into clear steps
4. Handle a variety of programming tasks more effectively

## Evaluating Slow-Thinking Abilities

There are two ways to evaluate the fine-tuned models:

### 1. REPAIRITY-Specific Evaluation

To evaluate specifically on reasoning capabilities with the held-out Claude test data:

```bash
python src/evaluation/repairity_eval.py --model_path [MODEL_PATH] --claude_test_path ./data/test/claude_traces_test_set_[STRATEGY].json
```

This script will:
1. Load the test set that wasn't used during training
2. Evaluate the model on these examples 
3. Save the results to a JSON file for analysis

### 2. Standard Benchmark Evaluation

To evaluate the model on standard BigCodeBench benchmarks:

```bash
python -m src.main evaluate --model qwen-7b --model_path ./models/qwen_repairity_[TIMESTAMP]_[STRATEGY] --output_dir ./eval_results_[STRATEGY]
```

For example:
```bash
# For the explicit reasoning model trained on 2023-03-30
python -m src.main evaluate --model qwen-7b --model_path ./models/qwen_repairity_20230330_123456_explicit --output_dir ./eval_results_explicit
```

This command will run the model through the standard evaluation pipeline and compare it against baseline models.

### Evaluation Metrics

When analyzing results, consider these aspects:
- Solution correctness
- Reasoning quality
- Problem-solving approach
- Edge case handling

## Technical Notes

- The script saves different models with suffixes corresponding to the reasoning strategy
- Training examples are saved in `./data/training_examples/` for inspection
- Test sets are saved in `./data/test/` for proper evaluation
- Each training run's configuration is saved in the model directory

## Troubleshooting

- **Memory Issues**: Reduce batch size or use fewer examples
- **Training Speed**: Adjust number of workers or use fewer strategies
- **Model Quality**: Try different reasoning strategies or collect more diverse examples
- **Evaluation**: If test set is too small, collect more data or use a different train/test split ratio 