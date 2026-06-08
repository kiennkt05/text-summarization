# Fine-tuning hyperparameters for PTM

# Unsloth/Qwen configurations
MODEL_NAME = "unsloth/Qwen2.5-1.5B"
MAX_SEQ_LENGTH = 2048 # Choose any! Unsloth auto supports RoPE Scaling internally!
MAX_NEW_TOKENS = 128 # Default max new tokens for generation
DTYPE = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = False # Use 4bit quantization to reduce memory usage. Set to False as per requirements.

# Hardware and paths
SEED = 3407
OUTPUT_DIR = "outputs_ptm"
