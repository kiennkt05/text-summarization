from unsloth import FastLanguageModel
import PTM.finetune.lora_config as lora_config

def apply_lora(model):
    """
    Applies LoRA adapters to the model.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_config.R,
        target_modules = lora_config.TARGET_MODULES,
        lora_alpha = lora_config.LORA_ALPHA,
        lora_dropout = lora_config.LORA_DROPOUT,
        bias = lora_config.BIAS,
        use_gradient_checkpointing = lora_config.USE_GRADIENT_CHECKPOINTING,
        random_state = lora_config.RANDOM_STATE,
        use_rslora = lora_config.USE_RSLORA,
        loftq_config = lora_config.LOFTQ_CONFIG,
    )
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
