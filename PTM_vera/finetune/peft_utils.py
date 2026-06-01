from unsloth import FastLanguageModel
import PTM_vera.finetune.vera_config as vera_config
from peft import VeraConfig, get_peft_model

def apply_vera(model):
    """
    Applies VeRA adapters to the model using peft's VeraConfig.
    """
    config = VeraConfig(
        r=vera_config.R,
        target_modules=vera_config.TARGET_MODULES,
        projection_prng_key=vera_config.PROJECTION_PRNG_KEY,
        save_projection=vera_config.SAVE_PROJECTION,
        vera_dropout=vera_config.VERA_DROPOUT,
        d_initial=vera_config.D_INITIAL,
        bias=vera_config.BIAS,
    )
    model = get_peft_model(model, config)
    return model

def save_model_adapters(model, tokenizer, save_path="lora_model"):
    """
    Saves the LoRA adapters locally.
    """
    model.save_pretrained(save_path) 
    tokenizer.save_pretrained(save_path)

def save_model_gguf(model, tokenizer, save_path="model", quantization_method="q4_k_m"):
    """
    Saves the model to GGUF format natively.
    """
    model.save_pretrained_gguf(save_path, tokenizer, quantization_method = quantization_method)
