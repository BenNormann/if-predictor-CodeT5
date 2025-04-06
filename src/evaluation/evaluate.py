"""
Evaluation module for CodeT5 if-statement prediction.
Handles model evaluation and metrics calculation.
"""

from typing import List, Dict, Any
import pandas as pd
import evaluate
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sys
import os
import subprocess
import tempfile

# Constants
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
BEAM_SIZE = 5
CODEBLEU_BLEU_WEIGHT = 0.7
CODEBLEU_EXACT_MATCH_WEIGHT = 0.3

def tokenize_data(
    data: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    max_length: int = MAX_INPUT_LENGTH
) -> Dict[str, torch.Tensor]:
    """
    Tokenize input data for model processing.
    
    @input:
        data (pd.DataFrame): Input data
        tokenizer (PreTrainedTokenizer): CodeT5 tokenizer
        max_length (int): Maximum sequence length
    
    @return:
        Dict[str, torch.Tensor]: Tokenized inputs
    """
    inputs = data['masked_code'].tolist()
    
    tokenized_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return tokenized_inputs

def calculate_sacrebleu(predictions: List[str], references: List[str]) -> float:
    """
    Calculate BLEU score using SacreBLEU implementation with smoothing
    to handle short sequences better.
    
    @input:
        predictions (List[str]): Predicted if conditions
        references (List[str]): Ground truth if conditions
    
    @return:
        float: SacreBLEU score (0-100)
    """
    # Format references for sacrebleu - each prediction needs its own list of references
    formatted_refs = [[ref] for ref in references]
    
    # Load the sacrebleu metric
    sacrebleu = evaluate.load("sacrebleu")
    
    # Calculate the SacreBLEU score with exponential smoothing
    # This is particularly good for short sequences
    results = sacrebleu.compute(
        predictions=predictions,
        references=formatted_refs,
        smooth_method='exp'  # Exponential smoothing is good for short sequences
    )
    
    return results["score"]

def calculate_exact_match_score(predictions: List[str], references: List[str]) -> float:
    """
    Calculate exact match ratio between predictions and references.
    This function is completely independent of any BLEU calculations.
    
    @input:
        predictions (List[str]): Predicted if conditions
        references (List[str]): Ground truth if conditions
    
    @return:
        float: Exact match ratio (0-100)
    """
    if not predictions or not references:
        return 0.0
    
    # Count exact matches
    match_count = 0
    total_count = min(len(predictions), len(references))
    
    for i in range(total_count):
        # Compare after stripping whitespace
        if predictions[i].strip() == references[i].strip():
            match_count += 1
    
    # Calculate percentage
    match_percentage = (match_count / total_count) * 100
    
    return match_percentage

def calculate_microsoft_codebleu(predictions: List[str], references: List[str], language: str = "python") -> float:
    """
    Simple direct call to Microsoft's CodeBLEU script.
    
    @input:
        predictions (List[str]): Predicted if conditions
        references (List[str]): List of reference conditions
        language (str): Programming language (default: python)
    
    @return:
        float: CodeBLEU score (0-100) or simplified score if script fails
    """
    # Write predictions and references to temporary files
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as hyp_file, \
         tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as ref_file:
        
        for pred in predictions:
            hyp_file.write(f"{pred}\n")
        for ref in references:
            ref_file.write(f"{ref}\n")
        
        hyp_path = hyp_file.name
        ref_path = ref_file.name
    
    try:
        # Direct call to the script exactly as shown in the example
        cmd = f"python calc_code_bleu.py --refs {ref_path} --hyp {hyp_path} --lang {language} --params 0.25,0.25,0.25,0.25"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Extract the score from the output
        for line in result.stdout.strip().split('\n'):
            if "CodeBLEU score:" in line:
                score = float(line.split()[-1])
                return score * 100  # Convert to 0-100 scale
        
        # Fallback if parsing fails
        print("Could not parse CodeBLEU output. Using simplified calculation.")
        return _calculate_simplified_codebleu(predictions, references)
    
    except Exception as e:
        print(f"Error running CodeBLEU: {e}")
        return _calculate_simplified_codebleu(predictions, references)
    
    finally:
        # Clean up temporary files
        os.remove(hyp_path)
        os.remove(ref_path)

def _calculate_simplified_codebleu(predictions: List[str], references: List[str]) -> float:
    """Calculate a simplified version of CodeBLEU using BLEU and exact match"""
    bleu_score = calculate_sacrebleu(predictions, references)
    exact_match = calculate_exact_match_score(predictions, references)
    return CODEBLEU_BLEU_WEIGHT * bleu_score + CODEBLEU_EXACT_MATCH_WEIGHT * exact_match

def generate_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: Dict[str, torch.Tensor],
    max_length: int = MAX_OUTPUT_LENGTH
) -> List[str]:
    """
    Generate predictions from model.
    
    @input:
        model (PreTrainedModel): Fine-tuned CodeT5 model
        tokenizer (PreTrainedTokenizer): CodeT5 tokenizer
        inputs (Dict[str, torch.Tensor]): Model inputs
        max_length (int): Maximum output length
    
    @return:
        List[str]: Generated if conditions
    """
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=BEAM_SIZE,
            early_stopping=True
        )
    
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return predictions

def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_data: pd.DataFrame,
    max_length: int = MAX_INPUT_LENGTH
) -> Dict[str, Any]:
    """
    Evaluate model on test dataset.
    
    @input:
        model (PreTrainedModel): Fine-tuned CodeT5 model
        tokenizer (PreTrainedTokenizer): CodeT5 tokenizer
        test_data (pd.DataFrame): Test dataset
        max_length (int): Maximum sequence length
    
    @return:
        Dict[str, Any]: Evaluation results including:
            - metrics: Dictionary of overall corpus-level metrics
            - predictions: List of model predictions
            - references: List of ground truth values
            - inputs: List of input functions
            - individual_results: List of individual metrics for each prediction
    """
    # Extract inputs and references
    inputs = test_data['masked_code'].tolist()
    references = test_data['target'].tolist()
    
    # Tokenize data
    tokenized_inputs = tokenize_data(test_data, tokenizer, max_length)
    if 'labels' in tokenized_inputs:
        del tokenized_inputs['labels']
    
    # Generate predictions
    predictions = generate_predictions(model, tokenizer, tokenized_inputs)
    
    # ------ Calculate overall corpus-level metrics ------
    corpus_metrics = {
        "sacrebleu": calculate_sacrebleu(predictions, references),
        "exact_match": calculate_exact_match_score(predictions, references),
        "codebleu": calculate_microsoft_codebleu(predictions, references)
    }
    
    # ------ Calculate individual metrics for each prediction ------
    individual_results = []
    
    # Load sacrebleu once for efficiency
    sacrebleu = evaluate.load("sacrebleu")
    
    for pred, ref in zip(predictions, references):
        # Check for exact match
        is_exact_match = pred.strip() == ref.strip()
        
        # Calculate BLEU score
        ref_list = [[ref]]
        bleu_result = sacrebleu.compute(
            predictions=[pred], 
            references=ref_list,
            smooth_method='exp'
        )
        bleu_score = bleu_result["score"]
        
        # Calculate CodeBLEU score
        try:
            codebleu_score = calculate_microsoft_codebleu([pred], [ref])
        except Exception:
            # Fallback if Microsoft CodeBLEU fails
            exact_match_value = 100 if is_exact_match else 0
            codebleu_score = CODEBLEU_BLEU_WEIGHT * bleu_score + CODEBLEU_EXACT_MATCH_WEIGHT * exact_match_value
        
        # Store individual results
        individual_results.append({
            "is_exact_match": is_exact_match,
            "bleu_score": bleu_score,
            "codebleu_score": codebleu_score
        })
    
    # ------ Compile all results ------
    result = {
        "metrics": corpus_metrics,
        "predictions": predictions,
        "references": references,
        "inputs": inputs,
        "individual_results": individual_results
    }
    
    return result

def save_results(
    results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save evaluation results to CSV file.
    Just formats and saves results, no calculations.
    
    @input:
        results (Dict[str, Any]): Evaluation results from evaluate_model
        output_path (str): Path to save results CSV
    """
    predictions = results["predictions"]
    references = results["references"]
    inputs = results["inputs"]
    individual_results = results["individual_results"]
    
    # Create formatted results for CSV
    formatted_results = []
    for i, (input_func, pred, ref, metrics) in enumerate(zip(
            inputs, predictions, references, individual_results)):
        
        pred_result = {
            "input_function": input_func,
            "is_exact_match": metrics["is_exact_match"],
            "expected_condition": ref,
            "predicted_condition": pred,
            "codebleu_score": metrics["codebleu_score"],
            "bleu_score": metrics["bleu_score"]
        }
        
        formatted_results.append(pred_result)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(formatted_results)
    
    # Reorder columns to match required format
    column_order = [
        "input_function",
        "is_exact_match",
        "expected_condition",
        "predicted_condition",
        "codebleu_score",
        "bleu_score"
    ]
    
    results_df = results_df[column_order]
    results_df.to_csv(output_path, index=False)
    
    print(f"Overall Metrics: {results['metrics']}")
    print(f"Results saved to {output_path}")

def main(test_data_path: str, model_path: str, output_path: str) -> None:
    """
    Main function to evaluate a model and save results.
    
    @input:
        test_data_path (str): Path to test data CSV
        model_path (str): Path to fine-tuned model
        output_path (str): Path to save results CSV
    """
    # Load test data
    test_data = pd.read_csv(test_data_path)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Evaluate model - all calculations done here
    results = evaluate_model(model, tokenizer, test_data)
    
    # Save results - no calculations, just formatting and saving
    save_results(results, output_path)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CodeT5 model on if-statement prediction")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--output", type=str, default="testset-results.csv", help="Path to save results CSV")
    
    args = parser.parse_args()
    main(args.test_data, args.model, args.output)

