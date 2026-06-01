from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import PTM.model.config as config

def get_trainer(
    args, model, tokenizer, train_dataset, eval_dataset=None
):
    """
    Initializes and returns the SFTTrainer.
    """
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            gradient_accumulation_steps = args.grad_accum,
            gradient_checkpointing = True,
            warmup_steps = args.warmup_steps,
            max_steps = args.max_steps,
            num_train_epochs = args.num_train_epochs,
            learning_rate = args.learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = args.logging_steps,
            optim = "adamw_8bit",
            weight_decay = args.weight_decay,
            lr_scheduler_type = "linear",
            seed = args.seed,
            output_dir = args.output_dir,
            report_to = "none", # Use this for WandB etc
            eval_strategy = "steps" if eval_dataset else "no",
            eval_steps = args.eval_steps if eval_dataset else None,
            save_strategy = "steps" if eval_dataset else "no",
            save_steps = args.eval_steps if eval_dataset else None,
            load_best_model_at_end = True if eval_dataset else False,
            metric_for_best_model = "loss" if eval_dataset else None,
        ),
    )
    return trainer
