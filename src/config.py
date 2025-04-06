"""
Configuration file for CodeT5 if-statement prediction project.
Contains constants used across the codebase.
"""

# =============================
# Data Processing Configuration
# =============================

NUM_SAMPLES = 1000 

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

MAX_CONDITION_LENGTH = 1000

MAX_FLATTENED_LENGTH = 1000

# =============================
# Model & Training Configuration
# =============================

DEFAULT_MODEL_NAME = "Salesforce/codet5-small"

DEFAULT_EPOCHS = 5

DEFAULT_BATCH_SIZE = 8
DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LOGGING_STEPS = 100

EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.01

MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 128

# =============================
# Evaluation Configuration
# =============================

CODEBLEU_BLEU_WEIGHT = 0.25
CODEBLEU_EXACT_MATCH_WEIGHT = 0.75

BEAM_SIZE = 3 