import torch
import unittest
from transformer.model.transformer import BaselineTransformer, Batch

class TestTransformer(unittest.TestCase):
    def test_forward_pass(self):
        """
        Verify that BaselineTransformer can compile and run a dummy forward pass.
        """
        vocab_size = 1000
        d_model = 64
        d_ff = 128
        h = 4
        N = 2
        
        model = BaselineTransformer(
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            N=N,
            vocab_size=vocab_size,
            dropout=0.1
        )
        
        batch_size = 4
        src_seq_len = 10
        tgt_seq_len = 8
        
        src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
        tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
        
        batch = Batch(src, tgt, pad_idx=0)
        
        output = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        
        self.assertEqual(output.shape, (batch_size, tgt_seq_len - 1, vocab_size))
        print("Forward pass successful! Output shape:", output.shape)

if __name__ == '__main__':
    unittest.main()
