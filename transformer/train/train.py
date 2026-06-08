import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import multiprocessing
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)


from ..model.transformer import BaselineTransformer, Batch
from transformer.train.scheduler import build_scheduler
from transformer.train import config
from utils.data.dataset import SummarizationDataset
from utils.data.collator import SummarizationCollator
from utils.data.tokenizer import load_tokenizer
from utils.data.preprocessing import segment_text

def seed(seed_value=42):
    """
    Ensure reproducibility across runs.
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    Seed worker processes for DataLoader reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_step(model, optimizer, criterion, scheduler, train_dataloader, pad_idx, epoch_num, scaler):
    """
    Execute a single training epoch.
    """
    model.train()
    device = next(model.parameters()).device
    device_type = device.type
    total_loss = 0
    train_bar = tqdm(train_dataloader, desc=f'Epoch {epoch_num} [TRAIN]')

    for src_ids, tgt_ids in train_bar:
        batch = Batch(src_ids, tgt_ids, pad_idx, device=device)
        optimizer.zero_grad()
        
        output = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = criterion(output.contiguous().view(-1, output.size(-1)), batch.tgt_y.contiguous().view(-1))
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        total_loss += loss.item()
        
        train_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(train_dataloader)

@torch.inference_mode()
def validate_step(model, criterion, validate_dataloader, pad_idx, epoch_num):
    """
    Validate model on the validation dataset.
    """
    model.eval()
    device = next(model.parameters()).device
    device_type = device.type
    total_loss = 0
    validate_bar = tqdm(validate_dataloader, desc=f'Epoch {epoch_num} [VALIDATE]')
    
    for src_ids, tgt_ids in validate_bar:
        batch = Batch(src_ids, tgt_ids, pad_idx, device=device)
        
        output = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = criterion(output.contiguous().view(-1, output.size(-1)), batch.tgt_y.contiguous().view(-1))
            
        total_loss += loss.item()
        validate_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(validate_dataloader)

def train_loop(model, optimizer, criterion, scheduler, train_dataloader, val_dataloader, pad_idx, epoch, best_val_loss=None, start_epoch=0, checkpoint_dir='checkpoints'):
    """
    Full training and validation loop with patience/early stopping.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('outputs/runs', exist_ok=True)
    
    patience = config.PATIENCE
    non_improve_count = 0
    best_val_loss = float('inf') if best_val_loss is None else best_val_loss
    is_cuda = torch.cuda.is_available()
    try:
        scaler = GradScaler(device='cuda', enabled=is_cuda)
    except TypeError:
        scaler = GradScaler(enabled=is_cuda)
    writer = SummaryWriter('outputs/runs/baseline_transformer')
    
    for e in range(start_epoch, start_epoch + epoch):
        train_epoch_loss = train_step(model, optimizer, criterion, scheduler, train_dataloader, pad_idx, e, scaler)
        val_epoch_loss = validate_step(model, criterion, val_dataloader, pad_idx, e)
        
        writer.add_scalar('Loss/train', train_epoch_loss, e)
        writer.add_scalar('Loss/val', val_epoch_loss, e)
        
        print(f"Epoch {e} Summary | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

        if val_epoch_loss < best_val_loss:
            non_improve_count = 0
            best_val_loss = val_epoch_loss
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }
            checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best checkpoint with val loss: {best_val_loss:.4f} to {checkpoint_path}")
        else:
            non_improve_count += 1
            print(f"Patience counter: {non_improve_count}/{patience}")

        if non_improve_count >= patience:
            print("Early stopping triggered!")
            break

    writer.close()

def main():
    """
    Main entry point for training from the command line.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Train custom Transformer from scratch")
    parser.add_argument("--train_path", type=str, help="Path to tokenized train parquet dataset")
    parser.add_argument("--val_path", type=str, help="Path to tokenized validation parquet dataset")
    parser.add_argument("--train_compounded_path", type=str, default=None, help="Optional path to save/load segmented train parquet")
    parser.add_argument("--val_compounded_path", type=str, default=None, help="Optional path to save/load segmented validation parquet")
    parser.add_argument("--tokenizer_path", type=str, default="train_summarization_tokenizer.json", help="Path to BPE tokenizer JSON file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    args = parser.parse_args()

    seed(config.SEED)

    # Load datasets
    print("Loading datasets...")
    
    # Optional compounded-parquet caching for train dataset
    if args.train_compounded_path and os.path.exists(args.train_compounded_path):
        print(f"Loading cached train dataset from {args.train_compounded_path}")
        train_df = pd.read_parquet(args.train_compounded_path)
    elif args.train_path:
        print(f"Loading raw train dataset from {args.train_path} and segmenting text...")
        train_df = pd.read_parquet(args.train_path)
        train_df = train_df.dropna()
        train_df['article'] = train_df['article'].parallel_apply(segment_text)
        train_df['summary'] = train_df['summary'].parallel_apply(segment_text)
        if args.train_compounded_path:
            print(f"Caching compounded train dataset to {args.train_compounded_path}")
            train_df.to_parquet(args.train_compounded_path)
    else:
        raise ValueError("No train data provided")
    
    # Optional compounded-parquet caching for validation dataset
    if args.val_compounded_path and os.path.exists(args.val_compounded_path):
        print(f"Loading cached validation dataset from {args.val_compounded_path}")
        val_df = pd.read_parquet(args.val_compounded_path)
    elif args.val_path:
        print(f"Loading raw validation dataset from {args.val_path} and segmenting text...")
        val_df = pd.read_parquet(args.val_path)
        val_df = val_df.dropna()
        val_df['article'] = val_df['article'].parallel_apply(segment_text)
        val_df['summary'] = val_df['summary'].parallel_apply(segment_text)
        if args.val_compounded_path:
            print(f"Caching compounded validation dataset to {args.val_compounded_path}")
            val_df.to_parquet(args.val_compounded_path)
    else:
        raise ValueError("No validation data provided")

    train_df = train_df.dropna()
    val_df = val_df.dropna()
    
    # Load tokenizer
    print("Loading tokenizer...")
    text_list = train_df['article'].to_list() + train_df['summary'].to_list()
    tokenizer = load_tokenizer(text_list=text_list, vocab_size=config.VOCAB_SIZE, save_path=args.tokenizer_path)
    pad_idx = tokenizer.token_to_id('<PAD>')
    bos_idx = tokenizer.token_to_id('<BOS>')
    eos_idx = tokenizer.token_to_id('<EOS>')

    # Tokenize datasets
    print("Tokenizing datasets...")
    train_df['article_ids'] = train_df['article'].apply(lambda x: tokenizer.encode(x).ids)
    train_df['summary_ids'] = train_df['summary'].apply(lambda x: tokenizer.encode(x).ids)
    val_df['article_ids'] = val_df['article'].apply(lambda x: tokenizer.encode(x).ids)
    val_df['summary_ids'] = val_df['summary'].apply(lambda x: tokenizer.encode(x).ids)

    # Replicate validation UNK-rate computation/printing from baseline-transformer.ipynb
    from torch.nn.utils.rnn import pad_sequence
    val_ids = [encoding.ids for encoding in tokenizer.encode_batch(val_df['article'].to_list() + val_df['summary'].to_list())]
    val_ids = pad_sequence([torch.tensor(ids) for ids in val_ids], batch_first=True, padding_value=pad_idx)
    count_unk = val_ids == tokenizer.token_to_id('<UNK>')
    padding_mask = val_ids == pad_idx
    print(count_unk.sum().item())
    print((count_unk.sum() / (count_unk.numel() - padding_mask.sum())).item())

    train_dataset = SummarizationDataset(train_df)
    val_dataset = SummarizationDataset(val_df)

    collator = SummarizationCollator(
        pad_token_id=pad_idx,
        bos_token_id=bos_idx,
        eos_token_id=eos_idx,
        max_seq_len=config.MAX_SEQ_LEN,
        max_sum_len=config.MAX_SUM_LEN
    )

    g = torch.Generator()
    g.manual_seed(config.SEED)
    
    optimal_workers = multiprocessing.cpu_count()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        persistent_workers=(optimal_workers > 0),
        num_workers=optimal_workers
    )

    validate_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=(optimal_workers > 0),
        num_workers=optimal_workers
    )

    print("Building model...")
    model = BaselineTransformer(
        d_model=config.D_MODEL,
        d_ff=config.D_FF,
        h=config.H,
        N=config.N,
        vocab_size=tokenizer.get_vocab_size(),
        dropout=config.DROPOUT
    )

    if torch.cuda.device_count() > 1:
        print(f"Activating DataParallel on {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(config.DEVICE)

    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    scheduler = build_scheduler(optimizer, d_model=config.D_MODEL, warmup_steps=config.WARMUP_STEPS)

    checkpoint_file = os.path.join(args.checkpoint_dir, 'best_checkpoint.pt')
    if os.path.exists(checkpoint_file):
        print(f"Found existing checkpoint at {checkpoint_file}. Resuming training...")
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        
        # Strip nn.DataParallel wrapper key if necessary
        model_state = checkpoint['model_state_dict']
        if isinstance(model, nn.DataParallel) and not any(k.startswith('module.') for k in model_state.keys()):
            model_state = {f'module.{k}': v for k, v in model_state.items()}
        elif not isinstance(model, nn.DataParallel) and any(k.startswith('module.') for k in model_state.keys()):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            
        model.load_state_dict(model_state)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        remain_epoch = args.epochs - start_epoch
        best_val_loss = checkpoint['best_val_loss']
        
        train_loop(
            model, optimizer, criterion, scheduler, 
            train_dataloader, validate_dataloader, pad_idx, 
            remain_epoch, best_val_loss=best_val_loss, 
            start_epoch=start_epoch, checkpoint_dir=args.checkpoint_dir
        )
    else:
        print("Starting training from scratch...")
        train_loop(
            model, optimizer, criterion, scheduler, 
            train_dataloader, validate_dataloader, pad_idx, 
            args.epochs, checkpoint_dir=args.checkpoint_dir
        )

if __name__ == '__main__':
    main()
