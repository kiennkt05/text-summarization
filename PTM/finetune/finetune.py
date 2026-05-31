import argparse
from PTM.model.model_loader import load_model_and_tokenizer
from PTM.finetune.peft_utils import apply_lora, save_model_adapters
from PTM.finetune.dataset import load_and_prepare_dataset
from PTM.finetune.trainer import get_trainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PTM using Unsloth")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training parquet dataset")
    parser.add_argument("--val_path", type=str, default=None, help="Path to validation parquet dataset")
    parser.add_argument("--save_path", type=str, default="outputs_ptm/lora_model", help="Path to save the LoRA adapters")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    print("Applying LoRA adapters...")
    model = apply_lora(model)

    print(f"Loading and preparing dataset from {args.train_path}...")
    train_dataset = load_and_prepare_dataset(args.train_path, tokenizer)
    
    val_dataset = None
    if args.val_path:
        print(f"Loading and preparing validation dataset from {args.val_path}...")
        val_dataset = load_and_prepare_dataset(args.val_path, tokenizer)

    print("Initializing trainer...")
    trainer = get_trainer(model, tokenizer, train_dataset, val_dataset, batch_size=args.batch_size, grad_accum=args.grad_accum)

    print("Starting training...")
    trainer_stats = trainer.train()
    
    print(f"Training completed. Runtime: {trainer_stats.metrics['train_runtime']} seconds.")
    
    print(f"Saving model to {args.save_path}...")
    save_model_adapters(model, tokenizer, save_path=args.save_path)
    print("Done!")

if __name__ == "__main__":
    main()
