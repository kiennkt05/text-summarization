import torch
from torch.utils.data import Dataset

class SummarizationDataset(Dataset):
    """
    Dataset for text summarization task.
    Returns tokenized article IDs and summary IDs.
    """
    def __init__(self, df, article_col='article_ids', summary_col='summary_ids'):
        self.df = df
        self.article_col = article_col
        self.summary_col = summary_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        article_ids = self.df.iloc[idx][self.article_col]
        summary_ids = self.df.iloc[idx][self.summary_col]
        return article_ids, summary_ids
