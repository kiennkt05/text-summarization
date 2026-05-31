import evaluate

def compute_bertscore(predictions, references):
    """
    Computes BERTScore for a set of predictions and references.
    Matches the BERTScore implementation in transformer/evaluate/metrics.py
    """
    bertscore = evaluate.load('bertscore')
    
    raw_bert = bertscore.compute(predictions=predictions, references=references, lang="vi")
    
    return {
        "bertscore": {
            "precision": sum(raw_bert["precision"]) / len(raw_bert["precision"]),
            "recall": sum(raw_bert["recall"]) / len(raw_bert["recall"]),
            "f1": sum(raw_bert["f1"]) / len(raw_bert["f1"])
        }
    }
