"""
Main training script for CodeT5 if-statement prediction.
Handles model training and fine-tuning process.
"""

import os
import pandas as pd
import subprocess
import warnings
from typing import Tuple
import sys
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)

# Add project root to path for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Use absolute imports
from src.evaluation.evaluate import evaluate_model, save_results
from src.data_processing.preprocess import tokenize_data
from src.config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_LOGGING_STEPS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    MAX_INPUT_LENGTH
)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

class IfConditionDataset(Dataset):
    """Dataset for if condition prediction."""
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings['input_ids'])

def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load and initialize CodeT5 model and tokenizer.
    
    @input:
        model_name (str): HuggingFace model name
    
    @return:
        Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]: Initialized model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    special_tokens = {'additional_special_tokens': ['<mask>']}
    tokenizer.add_special_tokens(special_tokens)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def setup_training_args(
    output_dir: str,
    num_train_epochs: int = DEFAULT_EPOCHS,
    per_device_train_batch_size: int = DEFAULT_BATCH_SIZE,
    per_device_eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    logging_steps: int = DEFAULT_LOGGING_STEPS
) -> TrainingArguments:
    """
    Set up training arguments for Trainer.
    
    @input:
        output_dir (str): Directory to save model checkpoints
        num_train_epochs (int): Number of training epochs
        per_device_train_batch_size (int): Training batch size
        per_device_eval_batch_size (int): Evaluation batch size
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
        logging_steps (int): Number of steps between logging
    
    @return:
        TrainingArguments: Configured training arguments
    """
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
        save_total_limit=1,
        push_to_hub=False,
        logging_steps=logging_steps,
        report_to="none",
        disable_tqdm=False,
    )

def train_model(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    training_args: TrainingArguments,
    output_dir: str
) -> AutoModelForSeq2SeqLM:
    """
    Train the model on the prepared dataset.
    
    @input:
        model (AutoModelForSeq2SeqLM): CodeT5 model
        tokenizer (AutoTokenizer): CodeT5 tokenizer
        train_data (pd.DataFrame): Training dataset
        val_data (pd.DataFrame): Validation dataset
        training_args (TrainingArguments): Training configuration
        output_dir (str): Directory to save model
    
    @return:
        AutoModelForSeq2SeqLM: Trained model
    """
    train_encodings = tokenize_data(train_data, tokenizer)
    val_encodings = tokenize_data(val_data, tokenizer)
    
    train_dataset = IfConditionDataset(train_encodings)
    val_dataset = IfConditionDataset(val_encodings)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )
    
    print("Starting training...")
    trainer.train()
    
    print("\nSaving model...")
    model.save_pretrained(os.path.join(output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
    
    return model

def prepare_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(file_path)

def main(
    data_dir: str = None,
    output_dir: str = None,
    model_name: str = DEFAULT_MODEL_NAME,
    num_train_epochs: int = DEFAULT_EPOCHS,
    preprocess: bool = False
) -> None:
    """
    Main training pipeline.
    
    @input:
        data_dir (str): Directory containing the dataset
        output_dir (str): Directory to save model and results
        model_name (str): HuggingFace model name
        num_train_epochs (int): Number of training epochs
        preprocess (bool): Whether to run preprocessing before training
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "saved_models")
        
    os.makedirs(output_dir, exist_ok=True)
    
    if preprocess:
        print("Running preprocessing...")
        preprocess_script = os.path.join(PROJECT_ROOT, "src", "data_processing", "preprocess.py")
        if os.path.exists(preprocess_script):
            subprocess.run(["python", preprocess_script], check=True, 
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Preprocessing complete.")
        else:
            print(f"Warning: Preprocess script not found at {preprocess_script}")
    
    print(f"Loading model and tokenizer from {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    print("Loading datasets...")
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train_data = prepare_dataset(train_path)
    val_data = prepare_dataset(val_path)
    test_data = prepare_dataset(test_path)
    
    print(f"Loaded {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test examples")
    
    print("Setting up training arguments...")
    training_args = setup_training_args(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs
    )
    
    print("Training model...\n")
    model = train_model(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        training_args=training_args,
        output_dir=output_dir
    )
    
    print("Evaluating model...")
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data
    )
    
    print("Saving results...")
    test_output_path = os.path.join(output_dir, "testset-results.csv")
    save_results(results, test_output_path)
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CodeT5 for if-statement prediction")
    parser.add_argument("--data_dir", type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "processed"), 
                        help="Data directory containing processed datasets")
    parser.add_argument("--output_dir", type=str, 
                        default=os.path.join(PROJECT_ROOT, "saved_models"), 
                        help="Output directory")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Model name")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing before training")
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.model_name, args.epochs, args.preprocess) 