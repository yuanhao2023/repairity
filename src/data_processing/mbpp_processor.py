from datasets import load_dataset

# Load MBPP dataset from Hugging Face
mbpp_dataset = load_dataset("mbpp")

# Print all categories in the dataset
dataset_categories = list(mbpp_dataset.keys())

# Print the number of entries in each category
dataset_sizes = {category: len(mbpp_dataset[category]) for category in dataset_categories}

# Print dataset details
for category in dataset_categories:
    print(f"Category: {category}, Size: {dataset_sizes[category]}")
    print(mbpp_dataset[category])
