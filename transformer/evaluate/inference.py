import torch
from transformer.model.transformer import Batch

@torch.inference_mode()
def generate_summary(model, src_token, src_mask, bos_idx, eos_idx, pad_idx, max_len=360):
    """
    Greedy search generator.
    Generates summary token IDs for a batch of input articles using the model's optimized KV-caching generate_summary method.
    """
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    return actual_model.generate_summary(src_token, src_mask, bos_idx, eos_idx, pad_idx, max_len=max_len)
