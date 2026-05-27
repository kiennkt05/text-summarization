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
pip install -e .
```

### 2. Running Verification Tests
To run unit tests verifying the model code compiles and runs a forward pass:
```bash
make test
```

### 3. Training the Transformer from Scratch
Run the resumable training script via command line:
```bash
python transformer/train/train.py \
  --train_path /path/to/train_compounded.parquet \
  --val_path /path/to/val_compounded.parquet \
  --tokenizer_path train_summarization_tokenizer.json
```
Or use the Makefile shortcut:
```bash
make train-scratch TRAIN_PATH=data/train.parquet VAL_PATH=data/val.parquet TOKENIZER_PATH=train_summarization_tokenizer.json
```
Training progress is automatically logged to TensorBoard and can be viewed using:
```bash
tensorboard --logdir outputs/runs
```
