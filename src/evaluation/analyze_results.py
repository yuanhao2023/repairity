#!/usr/bin/env python3
# analyze_results.py - Analyze and visualize evaluation results

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./evaluation_results",
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_analysis",
        help="Directory to save analysis results"
    )
    return parser.parse_args()

def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load results for all models."""
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json"):
            model_name = filename.replace("_results.json", "")
            with open(os.path.join(results_dir, filename), 'r') as f:
                results[model_name] = json.load(f)
    
    return results

def create_comparison_plots(results: Dict[str, Dict], output_dir: str):
    """Create comparison plots for the models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for comparison
    metrics_data = []
    for model_name, model_results in results.items():
        metrics = model_results["metrics"]
        metrics_data.append({
            "model": model_name,
            "total_problems": metrics["total_problems"],
            "problems_with_reasoning": metrics["problems_with_reasoning"],
            "avg_solution_length": metrics["avg_solution_length"]
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Set style
    plt.style.use('seaborn')
    
    # Create bar plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Problems with reasoning
    sns.barplot(data=df, x="model", y="problems_with_reasoning", ax=axes[0])
    axes[0].set_title("Problems with Reasoning")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Average solution length
    sns.barplot(data=df, x="model", y="avg_solution_length", ax=axes[1])
    axes[1].set_title("Average Solution Length")
    axes[1].set_ylabel("Characters")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Reasoning percentage
    df["reasoning_percentage"] = (df["problems_with_reasoning"] / df["total_problems"]) * 100
    sns.barplot(data=df, x="model", y="reasoning_percentage", ax=axes[2])
    axes[2].set_title("Percentage of Solutions with Reasoning")
    axes[2].set_ylabel("Percentage")
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()

def analyze_reasoning_patterns(results: Dict[str, Dict], output_dir: str):
    """Analyze patterns in reasoning approaches."""
    reasoning_patterns = {
        "step_by_step": 0,
        "here_is_my_reasoning": 0,
        "lets_think": 0,
        "first_we": 0,
        "approach": 0
    }
    
    # Only analyze RLLF model results
    rllf_results = results.get("rllf_qwen", {}).get("results", [])
    
    for result in rllf_results:
        solution = result["solution"].lower()
        if "step by step" in solution:
            reasoning_patterns["step_by_step"] += 1
        if "here's my reasoning" in solution:
            reasoning_patterns["here_is_my_reasoning"] += 1
        if "let's think" in solution:
            reasoning_patterns["lets_think"] += 1
        if "first we" in solution:
            reasoning_patterns["first_we"] += 1
        if "approach" in solution:
            reasoning_patterns["approach"] += 1
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(
        reasoning_patterns.values(),
        labels=reasoning_patterns.keys(),
        autopct='%1.1f%%'
    )
    plt.title("Distribution of Reasoning Patterns in RLLF Model")
    plt.savefig(os.path.join(output_dir, "reasoning_patterns.png"))
    plt.close()
    
    # Save pattern counts
    with open(os.path.join(output_dir, "reasoning_patterns.json"), 'w') as f:
        json.dump(reasoning_patterns, f, indent=2)

def create_summary_report(results: Dict[str, Dict], output_dir: str):
    """Create a summary report of the evaluation."""
    report = []
    report.append("# Model Evaluation Summary Report\n")
    
    # Add metrics comparison
    report.append("## Metrics Comparison\n")
    report.append("| Model | Total Problems | Problems with Reasoning | Avg Solution Length |")
    report.append("|-------|----------------|----------------------|-------------------|")
    
    for model_name, model_results in results.items():
        metrics = model_results["metrics"]
        report.append(
            f"| {model_name} | {metrics['total_problems']} | "
            f"{metrics['problems_with_reasoning']} | "
            f"{metrics['avg_solution_length']:.2f} |"
        )
    
    # Add improvement analysis
    report.append("\n## Improvement Analysis\n")
    
    # Calculate improvements
    initial_metrics = results["initial_qwen"]["metrics"]
    sft_metrics = results["sft_qwen"]["metrics"]
    rllf_metrics = results["rllf_qwen"]["metrics"]
    
    sft_improvement = (sft_metrics["problems_with_reasoning"] / initial_metrics["problems_with_reasoning"] - 1) * 100
    rllf_improvement = (rllf_metrics["problems_with_reasoning"] / sft_metrics["problems_with_reasoning"] - 1) * 100
    
    report.append(f"- SFT improved reasoning by {sft_improvement:.1f}% over initial model")
    report.append(f"- RLLF improved reasoning by {rllf_improvement:.1f}% over SFT model")
    
    # Save report
    with open(os.path.join(output_dir, "evaluation_summary.md"), 'w') as f:
        f.write("\n".join(report))

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    
    # Create visualizations
    create_comparison_plots(results, args.output_dir)
    analyze_reasoning_patterns(results, args.output_dir)
    create_summary_report(results, args.output_dir)

if __name__ == "__main__":
    main() 