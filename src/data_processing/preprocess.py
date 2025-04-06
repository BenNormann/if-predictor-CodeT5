"""
Data preprocessing module for CodeT5 if-statement prediction.
Handles dataset preparation, masking, and tokenization.
"""

import ast
import os
import json
import random
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from transformers import AutoTokenizer
from tqdm import tqdm
import sys

# Add the project root to path for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

# Import constants from config file using absolute import
from src import config

# Configuration parameters with absolute paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "datarawif.jsonl")  # Path to raw JSONL dataset
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")           # Output directory for processed data

class IfStatementVisitor(ast.NodeVisitor):
    """AST visitor to find if statements in Python code."""
    
    def __init__(self):
        self.if_statements = []
        
    def visit_If(self, node):
        # Store the if statement node with its condition and precise positions
        self.if_statements.append({
            'node': node,
            'condition': ast.unparse(node.test),
            'lineno': node.lineno,
            'col_offset': node.col_offset,
            'end_lineno': node.end_lineno,
            'end_col_offset': node.end_col_offset,
            'test_lineno': node.test.lineno,
            'test_col_offset': node.test.col_offset,
            'test_end_lineno': node.test.end_lineno,
            'test_end_col_offset': node.test.end_col_offset
        })
        
        # Continue traversing the AST
        self.generic_visit(node)

def extract_if_statements(code: str) -> List[Tuple[str, str]]:
    """
    Extract if statements from Python code and create masked versions.
    
    Args:
        code (str): Python function code
    
    Returns:
        List[Tuple[str, str]]: List of (flattened_original, flattened_masked) code examples
                              sorted by line number
    """
    results = []
    
    # Parse the code and find if statements
    if_statements = _find_if_statements(code)
    if not if_statements:
        return results
        
    # Get code lines for processing
    code_lines = code.split('\n')
    
    # Process each if statement in order of appearance
    for info in sorted(if_statements, key=lambda x: x['lineno']):
        try:
            example = _process_if_statement(info, code_lines)
            if example:
                results.append(example)
        except Exception as e:
            # Log the error but continue processing other statements
            print(f"Error processing if statement at line {info.get('lineno', 'unknown')}: {str(e)}")
            
    return results


def _find_if_statements(code: str) -> List[Dict[str, Any]]:
    """Parse code and extract if statement metadata."""
    try:
        tree = ast.parse(code)
        visitor = IfStatementVisitor()
        visitor.visit(tree)
        return visitor.if_statements
    except SyntaxError as e:
        print(f"Syntax error in code: {str(e)}")
        return []
    except Exception as e:
        print(f"Error parsing code: {str(e)}")
        return []


def _process_if_statement(info: Dict[str, Any], code_lines: List[str]) -> Optional[Tuple[str, str]]:
    """Process a single if statement and create masked version."""
    # Skip if condition is too complex or line is out of bounds
    if len(info['condition']) > config.MAX_CONDITION_LENGTH:
        return None
        
    line_idx = info['lineno'] - 1
    if line_idx >= len(code_lines):
        return None
        
    # Verify the line starts with "if "
    if_line = code_lines[line_idx]
    stripped_line = if_line.lstrip()
    if not stripped_line.startswith('if '):
        return None
        
    # Create masked line from AST info
    masked_line = create_masked_line_from_ast(if_line, code_lines, info)
    if not masked_line:
        return None
        
    # Extract complete if block including indented body
    indent_spaces = len(if_line) - len(stripped_line)
    if_block_lines = extract_if_block(code_lines, line_idx, indent_spaces)
    
    # Create masked version
    masked_lines = if_block_lines.copy()
    masked_lines[0] = masked_line
    
    # Flatten code and check length
    flattened_original = flatten_code('\n'.join(if_block_lines))
    flattened_masked = flatten_code('\n'.join(masked_lines))
    
    if len(flattened_masked) <= config.MAX_FLATTENED_LENGTH:
        return (flattened_original, flattened_masked)
    
    return None

def create_masked_line_from_ast(if_line: str, code_lines: List[str], info: Dict[str, Any]) -> str:
    """
    Create a masked version of the if line where "if condition" is replaced with "<mask>".
    
    Args:
        if_line: The original if line from the code
        code_lines: All lines of the source code
        info: Dictionary containing AST node information
        
    Returns:
        The masked version of the if line with "if condition" replaced by <mask>
    """
    # Extract key positions from AST
    if_node_col_offset = info['col_offset']  # Start of 'if' keyword
    before_if = if_line[:if_node_col_offset]
    
    # Check if condition is single-line or multi-line
    is_multiline = info['test_end_lineno'] > info['lineno']
    
    if is_multiline:
        return _mask_multiline_condition(before_if, code_lines, info)
    else:
        return _mask_singleline_condition(if_line, before_if, info)


def _mask_singleline_condition(if_line: str, before_if: str, info: Dict[str, Any]) -> str:
    """Helper function to mask a single-line if condition."""
    # Find text after the condition including the colon
    condition_end_col = info['test_end_col_offset']
    after_condition_text = if_line[condition_end_col:]
    colon_pos = after_condition_text.find(":")
    
    if colon_pos != -1:
        after_condition = after_condition_text[colon_pos:]  # Include colon and beyond
        return before_if + "<mask>" + after_condition
    else:
        # If no colon found (unusual), return the basic mask
        return before_if + "<mask>"


def _mask_multiline_condition(before_if: str, code_lines: List[str], info: Dict[str, Any]) -> str:
    """Helper function to mask a multi-line if condition."""
    # For multi-line, create a simple mask for the first line
    first_masked_line = before_if + "<mask>"
    
    # Try to find the colon which might be on a later line
    first_line_idx = info['lineno'] - 1
    last_line_idx = min(info['test_end_lineno'], len(code_lines)) - 1
    
    # Look for the line containing the colon
    for line_idx in range(first_line_idx, last_line_idx + 1):
        if line_idx >= len(code_lines):
            break
            
        if ":" in code_lines[line_idx]:
            # If colon is on the first line, include it after the mask
            if line_idx == first_line_idx:
                colon_pos = code_lines[line_idx].find(":")
                if colon_pos != -1:
                    first_masked_line = before_if + "<mask>" + code_lines[line_idx][colon_pos:]
            break
    
    return first_masked_line

def extract_if_block(code_lines: List[str], line_idx: int, block_indent: int) -> List[str]:
    """
    Extract the complete if block including its body.
    
    Args:
        code_lines: All lines of the source code
        line_idx: The index of the if statement line
        block_indent: The indentation level of the if statement
        
    Returns:
        List of code lines forming the if block (including the if line)
    """
    if_block_lines = [code_lines[line_idx]]
    i = line_idx + 1
    
    # Collect lines with greater indentation (part of if block)
    while i < len(code_lines):
        curr_line = code_lines[i]
        if not curr_line.strip():
            i += 1
            continue
            
        curr_indent = len(curr_line) - len(curr_line.lstrip())
        if curr_indent <= block_indent:
            break  # End of if block
            
        if_block_lines.append(curr_line)
        i += 1
        
    return if_block_lines

def flatten_code(code: str) -> str:
    """
    Flatten Python code by replacing indentation with <TAB> tokens.
    Preserves the first line exactly as is.
    
    Args:
        code (str): Python code with proper indentation
    
    Returns:
        str: Flattened code with <TAB> tokens
    """
    # Split code into lines without stripping
    lines = code.split('\n')
    
    # Remove empty lines but preserve non-empty whitespace lines
    lines = [line for line in lines if line.strip() != ""]
    
    if not lines:
        return ""
    
    # Get the first line with original indentation
    first_line = lines[0]
    base_indent = len(first_line) - len(first_line.lstrip())
    
    result = []
    
    # Keep first line exactly as is
    result.append(first_line.strip())
    
    # Add TAB tokens for subsequent lines
    for line in lines[1:]:
        if not line.strip():
            continue
            
        # Calculate the indentation level
        indent_level = len(line) - len(line.lstrip())
        
        # Calculate relative indentation (in tab units, assuming 4 spaces per tab)
        relative_tabs = max(0, (indent_level - base_indent) // 4)
        
        # Format the line with TAB tokens
        formatted_line = "<TAB> " * relative_tabs + line.lstrip()
        result.append(formatted_line)
    
    return " ".join(result)

def load_json_samples(data_path: str, num_samples: int = config.NUM_SAMPLES) -> List[Dict]:
    """Load JSON samples from JSONL file sequentially until we have enough valid samples."""
    # We'll load samples sequentially
    samples = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading samples from {data_path}"):
            if len(samples) >= num_samples * 3:  # Load 3x more to ensure we have enough after filtering
                break
                
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
                    
    print(f"Loaded {len(samples)} raw samples from {data_path}")
    return samples

def prepare_dataset_from_jsonl(data_path: str, num_samples: int = config.NUM_SAMPLES) -> pd.DataFrame:
    """Process dataset and extract if statements until we have exactly num_samples."""
    samples = load_json_samples(data_path, num_samples)
    
    data = {
        'target': [],
        'masked_code': [],
        'original_code': []
    }
    
    # Process samples until we have the exact number requested
    processed_count = 0
    for sample in tqdm(samples, desc="Processing functions"):
        # Break if we have enough examples
        if len(data['target']) >= num_samples:
            break
            
        processed_count += 1
        code = sample.get('content', '')
        if not code.strip():
            continue
        
        # Extract if statements
        if_statements = extract_if_statements(code)
        
        # Only take the first if statement from each code snippet
        if if_statements:
            original, masked = if_statements[0]
            data['target'].append(original)
            data['masked_code'].append(masked)
            data['original_code'].append(code)  # Add the original unfiltered code
    
    df = pd.DataFrame(data)
    
    # Ensure we don't exceed the requested number of samples
    if len(df) > num_samples:
        df = df.iloc[:num_samples]
    
    print(f"Processed {processed_count} samples to get {len(df)} valid examples")
    return df

def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets."""
    # Verify ratios sum to 1
    assert abs(config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO - 1.0) < 1e-10
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=config.RANDOM_SEED)
    
    # Calculate split indices
    n = len(df)
    train_end = int(config.TRAIN_RATIO * n)
    val_end = train_end + int(config.VAL_RATIO * n)
    
    # Split the dataset
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    
    print(f"Split dataset: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    return train_df, val_df, test_df

def prepare_and_save_splits() -> None:
    """Prepare dataset, split it, and save to CSV files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Prepare dataset
    df = prepare_dataset_from_jsonl(DATA_PATH, config.NUM_SAMPLES)
    print(f"Total examples extracted: {len(df)}")
    
    # Create a sample file with all columns (5 samples)
    sample_df = df.sample(n=min(5, len(df)), random_state=config.RANDOM_SEED)
    sample_df.to_csv(os.path.join(OUTPUT_DIR, 'sample.csv'), index=False, quoting=1)
    print(f"Saved sample with {len(sample_df)} examples to {os.path.join(OUTPUT_DIR, 'sample.csv')}")
    
    # Remove original_code from the main dataset
    df = df[['target', 'masked_code']]
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(df)
    
    # Save splits
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False, quoting=1)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False, quoting=1)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False, quoting=1)
    
    print(f"Saved datasets to {OUTPUT_DIR}")

# Public unified tokenization function to be used across the codebase
def tokenize_data(data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = config.MAX_INPUT_LENGTH) -> Dict[str, Any]:
    """
    Tokenize the masked and target code using the correct seq2seq approach.
    This function should be used consistently across training and evaluation.
    
    Args:
        data (pd.DataFrame): DataFrame containing 'masked_code' and 'target' columns
        tokenizer (AutoTokenizer): The tokenizer to use
        max_length (int): Maximum sequence length
        
    Returns:
        Dict[str, Any]: Dictionary containing tokenized inputs and labels
    """
    # Tokenize inputs (masked code)
    inputs = tokenizer(
        data['masked_code'].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    # Tokenize outputs (target code) with the special target tokenizer context
    with tokenizer.as_target_tokenizer():
        outputs = tokenizer(
            data['target'].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    
    inputs["labels"] = outputs.input_ids
    
    return inputs

if __name__ == "__main__":
    print("Starting data preparation...")
    prepare_and_save_splits() 