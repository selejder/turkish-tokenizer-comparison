"""Shared experiment settings.

The defaults intentionally describe a small CPU-friendly run. Increase
``TRAINING_STEPS`` and ``SAMPLE_CHARS`` for a longer experiment.
"""

SEED = 42

# Data and tokenizer settings
SAMPLE_CHARS = 10_000
NUM_MERGES = 100

# Model settings shared by both tokenizer experiments
BLOCK_SIZE = 64
BATCH_SIZE = 8
EMBED_DIM = 64
HEAD_SIZE = 16
NUM_HEADS = 4
NUM_TRANSFORMER_BLOCKS = 2

# Training and generation settings
LEARNING_RATE = 1e-3
TRAINING_STEPS = 20
EVALUATION_STEPS = 5
MAX_NEW_TOKENS = 100
PROMPT = "Yapay Zeka"
