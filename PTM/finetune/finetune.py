import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
from PTM.model.model_loader import load_model_and_tokenizer
from PTM.finetune.peft_utils import apply_lora, save_model_adapters
from PTM.finetune.dataset import load_and_prepare_dataset
from PTM.finetune.trainer import get_trainer
from PTM.model import config

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PTM using Unsloth")
    parser.add_argument("--model_name", type=str, default=config.MODEL_NAME, help="Name of the model to fine-tune")
    parser.add_argument("--max_seq_length", type=int, default=config.MAX_SEQ_LENGTH, help="Max sequence length")
    parser.add_argument("--dtype", type=str, default=config.DTYPE, help="Data type")
    parser.add_argument("--load_in_4bit", type=bool, default=config.LOAD_IN_4BIT, help="Load in 4bit")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR, help="Output directory")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training parquet dataset")
    parser.add_argument("--val_path", type=str, default=None, help="Path to validation parquet dataset")
    parser.add_argument("--save_path", type=str, default=f"{config.OUTPUT_DIR}/lora_model", help="Path to save the LoRA adapters")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Per device eval batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=60, help="Max training steps (overrides epochs if > 0)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluation and saving steps")
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args)

    print("Applying LoRA adapters...")
    model = apply_lora(model)

    print(f"Loading and preparing dataset from {args.train_path}...")
    train_dataset = load_and_prepare_dataset(args.train_path, tokenizer)
    
    val_dataset = None
    if args.val_path:
        print(f"Loading and preparing validation dataset from {args.val_path}...")
        val_dataset = load_and_prepare_dataset(args.val_path, tokenizer)

    print("Initializing trainer...")
    trainer = get_trainer(args, model, tokenizer, train_dataset, val_dataset)

    print("Starting training...")
    trainer_stats = trainer.train()
    
    print(f"Training completed. Runtime: {trainer_stats.metrics['train_runtime']} seconds.")
    
    print(f"Saving model to {args.save_path}...")
    save_model_adapters(model, tokenizer, save_path=args.save_path)
    print("Done!")

if __name__ == "__main__":
    main()
