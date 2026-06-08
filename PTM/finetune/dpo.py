import pandas as pd
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def formatting_prompts_func(examples):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Tóm tắt văn bản sau đây.

### Input:
{}

### Response:
"""
    
    articles = examples["article"]
    texts = []
    for article in articles:
        text = alpaca_prompt.format(article)
        texts.append(text)
    return { "prompt" : texts, }

def load_and_prepare_dataset(parquet_path, tokenizer):
    """
    Loads a dataset from a local parquet file and prepares it for fine-tuning.
    """
    df = pd.read_parquet(parquet_path)
    df = df[['article', 'references', 'predictions']]
    df.columns = ['article', 'chosen', 'rejected']
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: formatting_prompts_func(x), batched=True)
    return dataset

import argparse

def main():
    parser = argparse.ArgumentParser(description="DPO Fine-tuning script")
    parser.add_argument("--model_name", type=str, default="/kaggle/input/datasets/tkainguyen/qwen2-5-1-5blora/qwen2.5-0.5b", help="Path or name of the model")
    parser.add_argument("--dataset_path", type=str, default="/kaggle/input/datasets/tkainguyen/qwen2-5-1-5blora/RL_dataset.parquet", help="Path to the dataset parquet file")
    parser.add_argument("--output_dir", type=str, default="dpo_output", help="Output directory for the trained model")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for DPO")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Train batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--disable_dropout", action="store_true", help="Disable dropout in the model")
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # policy model: trainable
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # reference model: frozen copy
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    ref_model.requires_grad_(False)
    
    train_ds = load_and_prepare_dataset(args.dataset_path, tokenizer)
    
    config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        disable_dropout=args.disable_dropout,
    )
    
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=train_ds,
    )
    
    print(type(trainer.model))
    print(type(trainer.ref_model))
    
    trainable = 0
    all_params = 0
    
    for p in trainer.model.parameters():
        all_params += p.numel()
    
        if p.requires_grad:
            trainable += p.numel()
    
    print(f"trainable = {trainable:,}")
    print(f"all = {all_params:,}")
    print(f"ratio = {100*trainable/all_params:.4f}%")
    
    for name, p in trainer.model.named_parameters():
        if p.requires_grad:
            print(name)
            break
            
    trainer.train()

if __name__ == "__main__":
    main()
