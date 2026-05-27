import torch

# Model dimensions
D_MODEL = 512
D_FF = 2048
H = 8
N = 6
DROPOUT = 0.1

# Tokenizer & Padding
VOCAB_SIZE = 36000
MAX_SEQ_LEN = 1024
MAX_SUM_LEN = 360

# Training defaults
SEED = 42
WARMUP_STEPS = 4000
PATIENCE = 10

# Dynamic setups
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count()
BATCH_SIZE = 8 * max(1, NUM_GPUS)
EPOCHS = 100 if DEVICE == 'cuda' else 1