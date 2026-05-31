import torch
import evaluate
import json
from tqdm import tqdm
from transformer.model.transformer import Batch
from transformer.evaluate.inference import generate_summary
from utils.data.preprocessing import segment_text
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)

def evaluate_model(model, test_dataloader, bos_idx, eos_idx, pad_idx, tokenizer):
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    actual_model.eval()
    device = next(model.parameters()).device
      
    all_predictions = []
    all_references = []

    test_bar = tqdm(test_dataloader, desc='[GENERATING PREDICTIONS]')
    for src_ids, tgt_ids in test_bar:
        batch = Batch(src_ids, tgt_ids, pad_idx, device=device)
        prediction = actual_model.generate_summary(batch.src, batch.src_mask, bos_idx, eos_idx, pad_idx)
        prediction = tokenizer.decode_batch(prediction.cpu().tolist(), skip_special_tokens=True)
        references = tokenizer.decode_batch(batch.tgt_y.cpu().tolist(), skip_special_tokens=True)
        all_predictions.extend(prediction)
        all_references.extend(references)
        
    print(all_predictions[0])
    metrics_list = [
        evaluate.load('rouge'),
        evaluate.load('bleu'),
        evaluate.load('bertscore')
    ]

    final_results = {}

    for metric in metrics_list:
        print(f"{metric.name.upper()}")

        if metric.name in ["bertscore", "bert_score"]:
            raw_bert = metric.compute(predictions=all_predictions, references=all_references, lang="vi")
            final_results["bertscore"] = {
                "precision": sum(raw_bert["precision"]) / len(raw_bert["precision"]),
                "recall": sum(raw_bert["recall"]) / len(raw_bert["recall"]),
                "f1": sum(raw_bert["f1"]) / len(raw_bert["f1"])
            }
        else:
            final_results[metric.name] = metric.compute(predictions=all_predictions, references=all_references)

    with open("outputs/results.json", "w", encoding="utf-8") as f:
        # Saves both lists neatly paired in a single file
        json.dump({
            "predictions": all_predictions, 
            "references": all_references
        }, f, ensure_ascii=False, indent=2)
            
    return final_results

def main():
    """
    Main execution wrapper for running evaluations from CLI.
    """
    import argparse
    import os
    import pandas as pd
    from torch.utils.data import DataLoader
    from utils.data.dataset import SummarizationDataset
    from utils.data.collator import SummarizationCollator
    from utils.data.tokenizer import load_tokenizer
    from transformer.model.transformer import BaselineTransformer
    from transformer.train import config

    parser = argparse.ArgumentParser(description="Evaluate custom Transformer model")
    parser.add_argument("--test_path", type=str, help="Path to tokenized validation parquet dataset")
    parser.add_argument("--test_compounded_path", type=str, help="Path to tokenized compounded test dataset")
    parser.add_argument("--tokenizer_path", type=str, default="train_summarization_tokenizer.json", help="Path to BPE tokenizer JSON file")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_checkpoint.pt", help="Path to best model checkpoint")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(text_list=None, save_path=args.tokenizer_path)
    pad_idx = tokenizer.token_to_id('<PAD>')
    bos_idx = tokenizer.token_to_id('<BOS>')
    eos_idx = tokenizer.token_to_id('<EOS>')

    # Optional compounded-parquet caching for train dataset
    if args.test_compounded_path and os.path.exists(args.test_compounded_path):
        print(f"Loading cached train dataset from {args.test_compounded_path}")
        test_df = pd.read_parquet(args.test_compounded_path)
    elif args.test_path:
        print(f"Loading raw train dataset from {args.test_path} and segmenting text...")
        test_df = pd.read_parquet(args.test_path)
        test_df = test_df.dropna()
        test_df['article'] = test_df['article'].parallel_apply(segment_text)
        test_df['summary'] = test_df['summary'].parallel_apply(segment_text)
        if args.test_compounded_path:
            print(f"Caching compounded train dataset to {args.test_compounded_path}")
            test_df.to_parquet(args.test_compounded_path)
    else:
        raise ValueError("No test data provided")

    # Dropna and make sure column exists
    test_df = test_df.dropna()
    if 'article_ids' not in test_df.columns:
        print("Dataset not pre-tokenized. Tokenizing now...")
        test_df['article_ids'] = test_df['article'].apply(lambda x: tokenizer.encode(str(x)).ids)
        test_df['summary_ids'] = test_df['summary'].apply(lambda x: tokenizer.encode(str(x)).ids)

    test_dataset = SummarizationDataset(test_df)

    collator = SummarizationCollator(
        pad_token_id=pad_idx,
        bos_token_id=bos_idx,
        eos_token_id=eos_idx,
        max_seq_len=config.MAX_SEQ_LEN,
        max_sum_len=config.MAX_SUM_LEN
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
        num_workers=0
    )

    # Build and load model
    print("Building model...")
    model = BaselineTransformer(
        d_model=config.D_MODEL,
        d_ff=config.D_FF,
        h=config.H,
        N=config.N,
        vocab_size=config.VOCAB_SIZE,
        dropout=config.DROPOUT
    )

    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        checkpoint = torch.load(args.checkpoint_path, map_location=config.DEVICE)
        
        model_state = checkpoint['model_state_dict']
        if any(k.startswith('module.') for k in model_state.keys()):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            
        model.load_state_dict(model_state)
    else:
        print(f"[WARNING] Checkpoint not found at {args.checkpoint_path}. Evaluating untrained model...")

    model = model.to(config.DEVICE)

    # Run evaluation
    print("Running evaluation...")
    results = evaluate_model(
        model=model,
        test_dataloader=test_dataloader,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        tokenizer=tokenizer
    )

    print("\nEvaluation Results\n")
    print(f"|    Metric      | Score (%) |")
    print(f"| ROUGE-1        | {results['rouge']['rouge1']*100:.2f} |")
    print(f"| ROUGE-2        | {results['rouge']['rouge2']*100:.2f} |")
    print(f"| ROUGE-L        | {results['rouge']['rougeL']*100:.2f} |")
    print(f"| BLEU           | {results['bleu']['bleu']*100:.2f} |")
    print(f"| BERTScore (F1) | {results['bertscore']['f1']*100:.2f} |")

if __name__ == '__main__':
    main()