from unsloth import FastLanguageModel
import PTM_vera.model.config as config

def load_model_and_tokenizer(args):
    """
    Loads the pre-trained HuggingFace model and tokenizer using Unsloth's FastLanguageModel.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype = args.dtype,
        load_in_4bit = args.load_in_4bit,
        # token = "hf_...", # use one if using gated models
    )
    
    return model, tokenizer
