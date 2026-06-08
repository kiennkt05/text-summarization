# Unsloth returns the tokenizer directly alongside the model in FastLanguageModel.from_pretrained
# This file is intentionally left minimal, but can be used for tokenizer specific wrappers if needed later.

def get_special_tokens(tokenizer):
    """
    Helper function to get special tokens if needed.
    """
    return {
        "eos_token": tokenizer.eos_token,
        "bos_token": tokenizer.bos_token,
        "pad_token": tokenizer.pad_token,
    }
