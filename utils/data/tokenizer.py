import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def load_tokenizer(text_list, vocab_size=36000, save_path='train_summarization_tokenizer.json'):
    """
    Load a BPE tokenizer from save_path if it exists. Otherwise, train it from an iterator of texts and save.
    """
    if os.path.exists(save_path):
        print(f"Loading existing tokenizer from {save_path}")
        return Tokenizer.from_file(save_path)

    print(f"Tokenizer not found at {save_path}. Training new BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token='<UNK>'))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            '<PAD>',
            '<UNK>',
            '<BOS>',
            '<EOS>'
        ]
    )
    tokenizer.train_from_iterator(text_list, trainer)
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer