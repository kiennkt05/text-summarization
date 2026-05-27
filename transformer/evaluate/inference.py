import torch
from transformer.model.transformer import Batch

@torch.inference_mode()
def generate_summary(model, src_token, src_mask, bos_idx, eos_idx, pad_idx, max_len=1000):
    """
    Greedy search generator.
    Generates summary token IDs for a batch of input articles.
    """
    model.eval()
    device = next(model.parameters()).device
    device_type = device.type
    src_token = src_token.to(device)
    src_mask = src_mask.to(device)
    
    with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
        memory = model.encoder(src_token, src_mask)

    batch_size = src_token.size(0)
    tgt_token = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    unfinished = torch.ones((batch_size, 1), dtype=torch.bool, device=device)

    for _ in range(max_len):
        tgt_mask = Batch.make_std_mask(tgt_token, pad_idx)
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
            output = model.decoder(tgt_token, memory, src_mask, tgt_mask)

        next_token_logits = output[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1) # batch_size, 1

        # If a sequence in the batch is already finished, replace its generated tokens with padding
        next_token = next_token * unfinished + (~unfinished) * pad_idx
        tgt_token = torch.cat([tgt_token, next_token], dim=-1)

        unfinished = unfinished & (next_token != eos_idx)

        if unfinished.max() == 0:
            break
            
    return tgt_token
