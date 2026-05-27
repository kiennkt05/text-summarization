import torch
import evaluate
from tqdm import tqdm
from transformer.model.transformer import Batch
from transformer.evaluate.inference import generate_summary

def evaluate_model(model, test_dataloader, bos_idx, eos_idx, pad_idx, tokenizer, device='cpu'):
    """
    Evaluate the transformer on a test dataloader using ROUGE, BLEU, and BERTScore.
    """
    model.eval()
    model.to(device)

    all_predictions = []
    all_references = []

    test_bar = tqdm(test_dataloader, desc='[GENERATING PREDICTIONS]')
    for src_ids, tgt_ids in test_bar:
        batch = Batch(src_ids, tgt_ids, pad_idx, device=device)
        
        # Generate token IDs
        prediction_ids = generate_summary(model, batch.src, batch.src_mask, bos_idx, eos_idx, pad_idx)
        
        # Decode token IDs to text
        predictions = tokenizer.decode_batch(prediction_ids.cpu().tolist(), skip_special_tokens=True)
        references = tokenizer.decode_batch(batch.tgt_y.cpu().tolist(), skip_special_tokens=True)
        
        all_predictions.extend(predictions)
        all_references.extend(references)

    # Load HuggingFace evaluate metrics
    metrics_list = [
        evaluate.load('rouge'),
        evaluate.load('bleu'),
        evaluate.load('bertscore')
    ]

    final_results = {}

    for metric in metrics_list:
        print(f"Computing: {metric.name.upper()}...")

        if metric.name == "bertscore":
            raw_bert = metric.compute(predictions=all_predictions, references=all_references, lang="vi")
            final_results["bertscore"] = {
                "precision": sum(raw_bert["precision"]) / len(raw_bert["precision"]),
                "recall": sum(raw_bert["recall"]) / len(raw_bert["recall"]),
                "f1": sum(raw_bert["f1"]) / len(raw_bert["f1"])
            }
        else:
            final_results[metric.name] = metric.compute(predictions=all_predictions, references=all_references)

    return final_results
