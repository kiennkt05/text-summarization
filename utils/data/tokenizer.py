import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def load_tokenizer(text_list, vocab_size=36000, save_path='train_summarization_tokenizer.json'):
    """
    Train a BPE tokenizer from an iterator of texts and save it to disk.
    """
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
    tokenizer.save(save_path)
    return tokenizer
