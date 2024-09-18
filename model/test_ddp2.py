import os
import torch
import torch.nn as nn 
import tiktoken 
import argparse
import wandb 
from tqdm import tqdm 
from datasets import load_dataset
import torch.nn.functional as F 
from modeling_gpt2 import GPT2
from utils import tokenize, GPTDatasetV1, setup_seed
from torch.utils.data import DataLoader
from generate import text_to_token_ids, token_ids_to_text, generate
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Train gpt2 on sample-fineweb with DDP")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs")
    parser.add_argument('--vocab_size', type=int, default=50257, help="Tokenizer vocab size")
    parser.add_argument('--emb_dim', type=int, default=768, help="Embedding dimension")
    parser.add_argument('--num_layers', type=int, default=12, help="Number of transformer layers")
    parser.add_argument('--num_heads', type=int, default=12, help="Number of attention heads")
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Tokenizer name for tiktoken")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size per GPU")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument('--eval_interval', type=int, default=100, help="Evaluation interval in steps")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--max_length', type=int, default=512, help="Maximum sequence length")
    parser.add_argument('--data_name', type=str, default="eliplutchok/fineweb-small-sample", help="Dataset name")
    parser.add_argument('--sample_size', type=int, default=300000, help="Number of samples to use")
    parser.add_argument('--stride', type=int, default=256, help="Stride for dataset creation")
    return parser.parse_args()

def main(rank, world_size, args):
    setup(rank, world_size)
    setup_seed(args.seed + rank)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        wandb.init(project="gpt2-sample-fineweb-ddp", config=args)
    
    tokenizer = tiktoken.get_encoding(args.tokenizer_name)
    ds = load_dataset(args.data_name, split='train')
    ds = ds.select(range(min(args.sample_size, len(ds))))
    data = ds.train_test_split(0.3)

    train_data, val_data = data['train'], data['test']
    train_tokens = tokenize(tokenizer=tokenizer, data=train_data)
    val_tokens = tokenize(tokenizer=tokenizer, data=val_data)

    print("GPT-dataset")
    train_dataset = GPTDatasetV1(token_ids=train_tokens, max_length=args.max_length, stride=args.stride)
    val_dataset = GPTDatasetV1(token_ids=val_tokens, max_length=args.max_length, stride=args.stride)

    print("GPT-DistributedSampler")

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4)

    model = GPT2(args=args)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                            desc=f"GPU {rank} - Epoch {epoch+1}/{args.epochs}",
                            position=rank, leave=True)
        
        for step, (input_batch, target_batch) in progress_bar:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(input_batch)
            loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if rank == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1,
                    "step": step + 1
                })

            if (step + 1) % args.eval_interval == 0:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                
                with torch.no_grad():
                    for val_input_batch, val_target_batch in val_dataloader:
                        val_input_batch, val_target_batch = val_input_batch.to(device), val_target_batch.to(device)
                        val_logits = model(val_input_batch)
                        val_loss += F.cross_entropy(val_logits.flatten(0, 1), val_target_batch.flatten()).item()
                        val_steps += 1
                
                val_loss /= val_steps
                
                if rank == 0:
                    wandb.log({
                        "val_loss": val_loss,
                        "epoch": epoch + 1,
                        "step": step + 1
                    })
                    print(f"\nGPU {rank} - Step {step+1} - Train Loss: {train_loss/(step+1):.4f}, Validation Loss: {val_loss:.4f}")
                
                model.train()

        scheduler.step()

        if rank == 0:
            START_CONTEXT = "As an AI language model,"
            token_ids = generate(
                model=model.module,
                device=device,
                idx=text_to_token_ids(START_CONTEXT, tokenizer),
                max_new_tokens=20,
                context_len=args.max_length,
            )
            sample_text = token_ids_to_text(token_ids, tokenizer)
            print(f"\nSample text (GPU {rank}):", sample_text)
            
            wandb.log({
                "sample_text": sample_text,
                "epoch": epoch + 1
            })

    cleanup()

if __name__ == "__main__":
    args = parse_args()
    world_size = 2  # For two T4 GPUs on Kaggle
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)