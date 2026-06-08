# Vietnamese Text Summarization System

A modular, state-of-the-art framework for Vietnamese text summarization featuring:
1. A custom **Encoder-Decoder Transformer** built entirely from scratch in PyTorch.
2. A **Pretrained Model (PTM)** pipeline with parameter-efficient fine-tuning via **LoRA (PEFT)**.

---

## 📂 Project Structure

```text
text-summarization/
│
├── transformer/                   # Scratch Transformer Implementation
│   ├── model/
│   │   ├── attention.py           # ScaledDotProduct + MultiHeadAttention
│   │   ├── encoder.py             # EncoderLayer + Encoder
│   │   ├── decoder.py             # DecoderLayer + Decoder
│   │   ├── pos_encoding.py        # Sinusoidal PositionalEncoding
│   │   ├── ffn.py                 # PositionwiseFeedForward
│   │   ├── transformer.py         # Full Transformer architecture
│   │   └── beam_search.py         # [Placeholder] Beam search + length penalty
│   ├── train/
│   │   ├── train.py               # Main training loop (resumable)
│   │   ├── loss.py                # [Placeholder] LabelSmoothingLoss
│   │   ├── scheduler.py           # Noam Learning Rate Scheduler
│   │   └── config.py              # Hyperparameters (d_model, heads, etc.)
│   └── evaluate/
│       ├── rouge_eval.py          # [Placeholder] ROUGE-1/2/L calculation
│       ├── inference.py           # Single-sample / Greedy generation
│       └── metrics.py             # Extra metrics (ROUGE, BLEU, BERTScore)
│
├── PTM/                           # Pretrained Model Fine-tuning Pipeline
│   ├── model/
│   │   ├── model_loader.py        # [Placeholder] Load HuggingFace checkpoint
│   │   ├── config.py              # [Placeholder] Fine-tune hyperparams
│   │   └── tokenizer.py           # [Placeholder] Tokenizer wrapper
│   ├── finetune/
│   │   ├── finetune.py            # [Placeholder] Entry point for fine-tuning
│   │   ├── dataset.py             # [Placeholder] HuggingFace Dataset wrapper
│   │   ├── trainer.py             # [Placeholder] Custom Trainer / Arguments
│   │   ├── lora_config.py         # [Placeholder] LoRA / PEFT configuration
│   │   └── peft_utils.py          # [Placeholder] Adapter merge & save helpers
│   └── evaluate/
│       ├── rouge_eval.py          # [Placeholder] ROUGE evaluation
│       ├── generate.py            # [Placeholder] Batch generation script
│       ├── compare.py             # [Placeholder] Compare scratch vs pretrained
│       └── bertscore.py           # [Placeholder] BERTScore (bonus metric)
│
├── utils/                         # Reusable Data Pipelines & Utilities
│   ├── data/
│   │   ├── dataset.py             # Base PyTorch Dataset
│   │   ├── tokenizer.py           # BPE tokenizer training script
│   │   ├── preprocessing.py       # underthesea word-segmentation
│   │   ├── collator.py            # Sequence padding collate_fn
│   │   └── download.py            # [Placeholder] Google Drive downloader
│   └── notebooks/
│       ├── 01_eda.ipynb           # Exploratory data analysis
│       ├── 02_scratch.ipynb       # Scratch model development sandbox
│       ├── 03_finetune.ipynb      # Fine-tuning experiments
│       ├── 04_compare.ipynb       # Model evaluations comparison
│       └── 05_error_analysis.ipynb # Error cases analysis
│
├── configs/                       # Hyperparameter configurations
│   ├── scratch_base.yaml          # Scratch model parameters
│   └── finetune_lora.yaml         # LoRA PEFT parameters
│
├── outputs/                       # Saved summaries, plots, and TensorBoard logs
├── checkpoints/                   # Saved checkpoints (Gitignored)
├── tests/                         # Unit tests
├── requirements.txt               # Dependencies
├── setup.py                       # Project installer
├── Makefile                       # Training and validation shortcuts
└── README.md                      # Documentation
```

---

## 🚀 Getting Started

### 1. Installation
To install the system in development/editable mode (enabling clean absolute imports):
```bash
pip install -r requirements.txt
```

### 2. Running Verification Tests
To run unit tests verifying the model code compiles and runs a forward pass:
```bash
make test
```

### 3. Training the Transformer from Scratch
Run the resumable training script via command line:
```bash
python -m transformer.train.train \
  --train_compounded_path /path/to/train_compounded.parquet \
  --val_compounded_path /path/to/val_compounded.parquet \
  --tokenizer_path train_summarization_tokenizer.json
```

### 4. Evaluating the Model
Evaluate your checkpoints using the high-speed KV-caching decoder to compute ROUGE, BLEU, and BERTScore:
```bash
python -m transformer.evaluate.metrics \
  --test_path /path/to/test.parquet \
  --tokenizer_path train_summarization_tokenizer.json \
  --checkpoint_path checkpoints/best_checkpoint.pt
```

### 5. Fine-Tuning the Pretrained Model (PTM)
Run the Unsloth LoRA fine-tuning script via command line:
```bash
python -m PTM.finetune.finetune \
  --train_path /path/to/train_dataset.parquet \
  --val_path /path/to/val_dataset.parquet \
  --save_path outputs_ptm/lora_model
```

### 6. Evaluating the Pretrained Model
Evaluate your LoRA checkpoints on a test dataset to compute ROUGE, BLEU, and BERTScore using native 2x faster Unsloth inference:
```bash
python -m PTM.evaluate.generate \
  --test_path /path/to/test.parquet \
  --model_path outputs_ptm/lora_model
```
*(Note: If `--model_path` is omitted or invalid, it will automatically fall back to evaluating the original pretrained model.)*

To generate a summary for a single text, replace `--test_path` with `--text`:
```bash
python -m PTM.evaluate.generate \
  --model_path outputs_ptm/lora_model \
  --text "Văn bản tiếng Việt cần tóm tắt..."
```
### 7. Preparing Dataset for DPO
Before running DPO, generate the RL dataset by scoring predictions with the custom reward function:
```bash
python -m PTM.finetune.prepare_rl_dataset \
  --input_path /content/lora_train_results.json \
  --full_output_path full_RL_dataset.parquet \
  --worst_output_path RL_dataset.parquet \
  --top_k_ratio 0.25
```

### 8. Direct Preference Optimization (DPO) Fine-Tuning
For further alignment using human/AI preferences, run the DPO fine-tuning script:
```bash
python -m PTM.finetune.dpo \
  --model_name /kaggle/input/datasets/tkainguyen/qwen2-5-1-5blora/qwen2.5-0.5b \
  --dataset_path RL_dataset.parquet \
  --output_dir dpo_output \
  --beta 0.1 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --disable_dropout
```

### 9. LLM as a Judge Evaluation
To comprehensively evaluate multiple models side-by-side using Gemini as a judge (G-Eval approach), ensure you have set `GEMINI_API_KEY` in your `.env` file, then run:
```bash
python -m PTM.evaluate.llm_as_judge \
  --input_path /content/valid-sampled-predictions.parquet \
  --output_path llm_judge_results.csv \
  --model gemini-3.1-flash-lite
```