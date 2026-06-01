import evaluate

def compute_rouge_and_bleu(predictions, references):
    """
    Computes ROUGE and BLEU metrics for a set of predictions and references.
    Matches the metrics implemented in transformer/evaluate/metrics.py
    """
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    
    rouge_res = rouge.compute(predictions=predictions, references=references)
    bleu_res = bleu.compute(predictions=predictions, references=references)
    
    return {
        "rouge": rouge_res,
        "bleu": bleu_res
    }
