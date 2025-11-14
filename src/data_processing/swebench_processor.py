from datasets import load_dataset
import pandas as pd
import random
import textwrap
import os
import subprocess
import tempfile
import shutil

# Load the SWE-bench Verified dataset
dataset = load_dataset("princeton-nlp/SWE-bench_Verified")

def clone_repo_at_commit(repo_url, commit_sha, temp_dir):
    """Clone a repository at a specific commit and return the path."""
    # Extract repo name from URL for the folder name
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    repo_path = os.path.join(temp_dir, repo_name)
    
    print(f"Cloning repository: {repo_url} at commit {commit_sha}")
    
    # Clone the repository
    subprocess.run(['git', 'clone', repo_url, repo_path], check=True)
    
    # Checkout the specific commit
    subprocess.run(['git', 'checkout', commit_sha], cwd=repo_path, check=True)
    
    return repo_path

def display_repo_content(repo_url, commit_sha):
    """Display the content of a repository at a specific commit."""
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            repo_path = clone_repo_at_commit(repo_url, commit_sha, temp_dir)
            
            # List the top-level directories and files
            print(f"\nRepository structure at commit {commit_sha}:")
            for root, dirs, files in os.walk(repo_path, topdown=True, onerror=None, followlinks=False):
                level = root.replace(repo_path, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                sub_indent = ' ' * 4 * (level + 1)
                
                # Limit the depth and number of files shown to avoid overwhelming output
                if level > 2:  # Only go 3 levels deep
                    dirs[:] = []
                    continue
                
                file_count = 0
                for file in files:
                    if file_count < 5:  # Only show up to 5 files per directory
                        print(f"{sub_indent}{file}")
                        file_count += 1
                    else:
                        remaining = len(files) - 5
                        if remaining > 0:
                            print(f"{sub_indent}... and {remaining} more files")
                        break
                        
                # Limit the number of directories shown at each level
                if len(dirs) > 5:
                    dirs[:] = dirs[:5]
                    print(f"{sub_indent}... and {len(dirs) - 5} more directories")
                
        except Exception as e:
            print(f"Error accessing repository: {e}")

def display_example(example, show_repo_content=False):
    """Display a formatted example with repository content, problem statement, and patch."""
    repo = example['repo']
    commit_sha = example.get('base_commit', 'N/A')
    
    print("\n" + "="*80)
    print(f"Repository: {repo}")
    print(f"Commit SHA: {commit_sha}")
    print(f"Instance ID: {example['instance_id']}")
    print(f"Difficulty: {example['difficulty']}")
    
    # Display problem statement
    print("\nPROBLEM STATEMENT:")
    wrapped_problem = textwrap.fill(str(example['problem_statement']), width=80)
    print(wrapped_problem)
    
    # Display patch if available
    if 'patch' in example and example['patch']:
        print("\nPATCH:")
        wrapped_patch = textwrap.fill(str(example['patch']), width=80)
        print(wrapped_patch)
    
    # Show repository content if requested
    if show_repo_content and commit_sha != 'N/A':
        # Convert repo name to GitHub URL
        # Assuming format like "pytorch/pytorch"
        github_url = f"https://github.com/{repo}.git"
        display_repo_content(github_url, commit_sha)
    
    print("="*80)

# 1. Basic dataset information
print(f"Dataset structure: {dataset}")
print(f"Number of examples: {len(dataset['test'])}")

# 2. Get repository and commit information
repos = {}
repo_commits = {}

for item in dataset['test']:
    repo = item['repo']
    commit_sha = item.get('base_commit', 'N/A')
    
    repos[repo] = repos.get(repo, 0) + 1
    if repo not in repo_commits:
        repo_commits[repo] = commit_sha

print(f"\nUnique repositories: {len(repos)}")
print("Top 5 repositories by example count:")
for repo, count in sorted(repos.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {repo}: {count} examples, Commit: {repo_commits[repo]}")

# 3. Show all repositories with commit SHAs
print("\nAll repositories in the dataset with commit SHAs:")
for repo, count in sorted(repos.items(), key=lambda x: x[0]):
    print(f"  {repo}: {count} examples, Commit: {repo_commits[repo]}")

# 4. Show detailed examples with repository content, problem statement, and patch
print("\n\nDETAILED EXAMPLES WITH REPOSITORY CONTENT:")

# Get a few examples from different repositories (up to 3)
sample_repos = list(repos.keys())[:3]  # Taking first 3 repos for demonstration
for repo in sample_repos:
    # Find an example from this repository
    for example in dataset['test']:
        if example['repo'] == repo:
            print(f"\n\nEXAMPLE FROM REPOSITORY: {repo}")
            # Show detailed information including repository content
            display_example(example, show_repo_content=True)
            break

print("\nNOTE: To clone and examine all repositories or more examples would require more time and resources.")
print("This script provides a sample of the repository structure and content.")