# LoRA/PEFT configurations

# Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
R = 16 
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_ALPHA = 16
LORA_DROPOUT = 0 # Supports any, but = 0 is optimized
BIAS = "none"    # Supports any, but = "none" is optimized

# [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
USE_GRADIENT_CHECKPOINTING = "unsloth" # True or "unsloth" for very long context
RANDOM_STATE = 3407
USE_RSLORA = False  # We support rank stabilized LoRA
LOFTQ_CONFIG = None # And LoftQ
