# If-Statement Condition Predictor with CodeT5

* [1. Introduction](#1-introduction)  
* [2. Getting Started](#2-getting-started)  
* [3. Running Options](#3-running-options)
* [4. Results](#4-results)  

## 1. Introduction
This project implements a **fine-tuned CodeT5 model** for if-statement condition prediction in Python code. The model learns to predict the missing condition in an if statement by training on Python code examples.

## 2. Getting Started

### Prerequisites
- Python 3.9+
- Required packages: torch, transformers, datasets, evaluate, pandas, numpy, sacrebleu, ast, tqdm, scikit-learn

### Data Download
The dataset files are too large to be included in this repository. Please download them separately:

1. Download the `datarawif.zip` or `datarawif.7z` file from [this Google Drive link](#) (replace with your actual download link)
2. Place the file in the project root directory

### Installation
```bash
# Clone repository
git clone https://github.com/BenNormann/if-predictor-CodeT5
cd if-predictor-CodeT5

# Install dependencies
pip install -r requirements.txt

# Extract dataset
mkdir -p data/raw
unzip datarawif.zip -d data/raw
# OR if using 7z file:
# 7z x datarawif.7z -odata/raw
```

The dataset should be extracted as `data/raw/datarawif.jsonl`.

## 3. Running Options

### Preprocessing

Before training the model, you need to preprocess the raw data to extract if-statements and create the training examples:

```bash
# Create directory structure for processed data
mkdir -p data/processed

# Run preprocessing script
python src/data_processing/create_filtered_dataset.py
```

This preprocessing will:
- Parse Python code from the raw dataset
- Extract if-statements and their conditions
- Create masked versions by replacing if conditions with `<mask>` tokens
- Generate input-output pairs for training
- Split the data into train, validation, and test sets (80%/10%/10%)
- Save the processed datasets as CSV files in the `data/processed` directory

### Training

To train the model with default settings:
```bash
python src/train.py
```

The default settings will:
- Use the processed data in the default location
- Save the model to the default output directory (saved_models)
- Use the Salesforce/codet5-small model
- Train for 5 epochs with standard hyperparameters

### Advanced Options

If you need to customize training, the following parameters are available:

```bash
--data_dir PATH           # Path to the processed data directory
--output_dir PATH         # Directory to save the trained model and results
--model_name STRING       # HuggingFace model name (default: Salesforce/codet5-small)
--num_train_epochs INT    # Number of training epochs (default: 5)
--batch_size INT          # Training batch size (default: 8)
--eval_batch_size INT     # Evaluation batch size (default: 8)
--learning_rate FLOAT     # Learning rate (default: 5e-5)
--weight_decay FLOAT      # Weight decay (default: 0.01)
```

## 4. Results

After training, the following results will be saved to the specified output directory:

- The trained model checkpoint in the `best_model` directory
- Evaluation metrics in the `results.json` file including:
  - BLEU scores
  - CodeBLEU scores 
  - Exact match accuracy
  - Prediction samples

You can evaluate the model on the test set using:

```bash
python src/evaluation/evaluate.py --model_dir saved_models/best_model
```

Lower perplexity values and higher CodeBLEU scores indicate better model performance.
