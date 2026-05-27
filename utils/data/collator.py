import torch
from torch.nn.utils.rnn import pad_sequence

class SummarizationCollator:
    """
    Collate function wrapper to pad article and summary sequences and append BOS/EOS tokens.
    """
    def __init__(self, pad_token_id, bos_token_id, eos_token_id, max_seq_len=1024, max_sum_len=360):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_seq_len = max_seq_len
        self.max_sum_len = max_sum_len

    def __call__(self, batch):
        article_tensors = []
        summary_tensors = []
        for article_ids, summary_ids in batch:
            # Truncate article sequences to max_seq_len
            article_tensors.append(torch.tensor(article_ids[:self.max_seq_len]))

            # Format summary: BOS + summary_tokens (truncated) + EOS
            truncated_sum = summary_ids[:self.max_sum_len - 2]
            summary_ids_formatted = [self.bos_token_id] + truncated_sum + [self.eos_token_id]
            summary_tensors.append(torch.tensor(summary_ids_formatted))

        # Pad sequences in batch
        article_tensors = pad_sequence(article_tensors, batch_first=True, padding_value=self.pad_token_id)
        summary_tensors = pad_sequence(summary_tensors, batch_first=True, padding_value=self.pad_token_id)
        
        return article_tensors, summary_tensors
