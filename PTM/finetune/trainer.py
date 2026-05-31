from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import PTM.model.config as config

def get_trainer(model, tokenizer, train_dataset, eval_dataset=None, batch_size=2, grad_accum=4):
    """
    Initializes and returns the SFTTrainer.
    """
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = config.MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            
            # Use num_train_epochs = 1, warmup_ratio for full training runs!
            warmup_steps = 5,
            max_steps = 60, # NOTE: Increase num_train_epochs for full run and turn off max_steps=None

            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = config.SEED,
            output_dir = config.OUTPUT_DIR,
            report_to = "none", # Use this for WandB etc
            evaluation_strategy = "steps" if eval_dataset else "no",
            eval_steps = 10 if eval_dataset else None,
            save_strategy = "steps" if eval_dataset else "no",
            save_steps = 10 if eval_dataset else None,
            load_best_model_at_end = True if eval_dataset else False,
            metric_for_best_model = "loss" if eval_dataset else None,
        ),
    )
    return trainer
