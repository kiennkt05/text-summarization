from unsloth import FastLanguageModel
import PTM.model.config as config

def load_model_and_tokenizer():
    """
    Loads the pre-trained HuggingFace model and tokenizer using Unsloth's FastLanguageModel.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.MODEL_NAME,
        max_seq_length = config.MAX_SEQ_LENGTH,
        dtype = config.DTYPE,
        load_in_4bit = config.LOAD_IN_4BIT,
        # token = "hf_...", # use one if using gated models
    )
    
    return model, tokenizer
