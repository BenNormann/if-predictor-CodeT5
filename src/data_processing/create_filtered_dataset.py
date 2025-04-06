"""
Filter raw dataset to keep only functions with if statements.
Creates datarawif.jsonl from dataraw.jsonl.
"""

import os
import json
import sys
import re
from tqdm import tqdm

# Add the project root to path for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

# Source and target paths
SOURCE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "dataraw.jsonl")
TARGET_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "datarawif.jsonl")

def has_if_statement(code):
    """Check if a code sample contains an if statement using regex."""
    return bool(re.search(r'\bif\s', code))

def filter_dataset():
    """Filter the dataset to keep only functions with if statements."""
    # Count total lines first
    with open(SOURCE_PATH, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Processing {total_lines} samples from {SOURCE_PATH}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(TARGET_PATH), exist_ok=True)
    
    # Process each line and filter
    kept_count = 0
    with open(SOURCE_PATH, 'r', encoding='utf-8') as source_file, \
         open(TARGET_PATH, 'w', encoding='utf-8') as target_file:
        
        for line in tqdm(source_file, total=total_lines, desc="Filtering functions"):
            try:
                sample = json.loads(line)
                code = sample.get('content', '')
                
                if code.strip() and has_if_statement(code):
                    # Keep this sample
                    target_file.write(line)
                    kept_count += 1
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue
    
    # Print statistics
    filtered_count = total_lines - kept_count
    print(f"Filtered out {filtered_count} samples without if statements ({filtered_count/total_lines:.2%})")
    print(f"Kept {kept_count} samples with if statements ({kept_count/total_lines:.2%})")
    print(f"Filtered dataset saved to {TARGET_PATH}")

if __name__ == "__main__":
    print("Starting dataset filtering...")
    filter_dataset()
    print("Filtering complete!") 