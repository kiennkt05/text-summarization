import pandas as pd
from datasets import Dataset

def formatting_prompts_func(examples, tokenizer):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Tóm tắt văn bản sau đây.

### Input:
{}

### Response:
{}"""
    
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    articles = examples["article"]
    summaries = examples["summary"]
    texts = []
    for article, summary in zip(articles, summaries):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(article, summary) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def load_and_prepare_dataset(parquet_path, tokenizer):
    """
    Loads a dataset from a local parquet file and prepares it for fine-tuning.
    """
    df = pd.read_parquet(parquet_path)
    # Drop rows where article or summary might be missing
    df = df.dropna(subset=['article', 'summary'])
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: formatting_prompts_func(x, tokenizer), batched = True,)
    return dataset
