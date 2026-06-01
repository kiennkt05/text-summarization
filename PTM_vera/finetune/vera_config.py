# VeRA/PEFT configurations

# Choose any number > 0 ! Suggested 256 for VeRA since it's highly parameter efficient
R = 256 
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
PROJECTION_PRNG_KEY = 0
SAVE_PROJECTION = True
VERA_DROPOUT = 0.0
D_INITIAL = 0.1
BIAS = "none"

# Use gradient checkpointing (True or "unsloth" depending on model, standard peft accepts boolean generally, but we can pass it if Unsloth is base)
# We will use standard boolean here, as peft expects it
USE_GRADIENT_CHECKPOINTING = True
