import os
import torch
import torch.nn as nn
import numpy as np 
import tiktoken
import argparse
import wandb
from tqdm import tqdm
from datasets import load_dataset
import torch.nn.functional as F
from modeling_gpt2 import GPT2
from utils import tokenize, GPTDatasetV1, setup_seed, save_model_checkpoint
from torch.utils.data import DataLoader
from generate import text_to_token_ids, token_ids_to_text, generate
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 on sample-fineweb with DDP")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs")
    parser.add_argument('--train_data', type=str, help="path to the train npz file")
    parser.add_argument('--test_data', type=str,  help="path to the test npz file")
    parser.add_argument('--eval_batch_size', default=4 , type=int,  help="batch size of 4 for eval")
    parser.add_argument('--context_len', type=int, default=20,help="max length for the model generatng text")
    parser.add_argument('--vocab_size', type=int, default=50257, help="Tokenizer vocab size")
    parser.add_argument('--emb_dim', type=int, default=684, help="Embedding dimension")
    parser.add_argument('--num_layers', type=int, default=6, help="Number of transformer layers")
    parser.add_argument('--num_heads', type=int, default=12, help="Number of attention heads")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size per GPU")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument('--eval_interval', type=int, default=100, help="Evaluation interval in steps")
    parser.add_argument('--save_step', type=int, default=2500, help="what step should we save the model ")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--max_length', type=int, default=512, help="Maximum sequence length")
    parser.add_argument('--data_name', type=str, default="eliplutchok/fineweb-small-sample", help="Dataset name")
    parser.add_argument('--sample_size', type=int, default=300000, help="Number of samples to use")
    parser.add_argument('--stride', type=int, default=256, help="Stride for dataset creation")
    parser.add_argument('--world_size', type=int, default=2, help="Number of GPUs (DDP)")
    return parser.parse_args()

def main(rank, args):
    try:
        setup(rank, args.world_size)

        # Set up device
        device = torch.device(f"cuda:{rank}")

        # Initialize wandb only for rank 0
        wandb.login(key="04098c64a0b88d5f4ff90335b7f75613041420c6")
        if rank == 0:
            wandb.init(project="gpt2-sample-fineweb-ddp", config=args, group=f"DDP_GPT2",)

        
        # Load and preprocess data
        tokenizer = tiktoken.get_encoding("gpt2")
   
        train_dataset = GPTDatasetV1(args.train_data)
        val_dataset = GPTDatasetV1(args.test_data)

        # Set up distributed data loaders and samplers
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, sampler=val_sampler, num_workers=4)

        # Initialize model
        model = GPT2(args)
        model.to(device)
        model = DDP(model, device_ids=[rank])
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params:,}")
        
        # Set up optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = int(0.1 * num_training_steps)  
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)          
        scaler = torch.GradScaler()

        for epoch in range(args.epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            train_loss = 0.0
            
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Rank {rank} Epoch {epoch+1}/{args.epochs}", leave=True)
            
            for step, (input_batch, target_batch) in enumerate(train_dataloader):
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_batch)
                    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

                scheduler.step()

                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
                if rank == 0:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "epoch": epoch + 1,
                        "step": step + 1,
                    })



                if (step + 1) % args.eval_interval == 0 and rank == 0:  # Only rank 0 evaluates
                    model.eval()
                    val_loss = 0.0
                    val_steps = 0
                    
                    with torch.no_grad():
                        val_progress_bar = tqdm(val_dataloader, desc=f"Validating (Rank {rank})", leave=False)
                        for val_input_batch, val_target_batch in val_progress_bar:
                            val_input_batch, val_target_batch = val_input_batch.to(device), val_target_batch.to(device)
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                val_logits = model(val_input_batch)
                                batch_loss = F.cross_entropy(val_logits.flatten(0, 1), val_target_batch.flatten()).item()
                                val_loss += batch_loss
                            val_steps += 1
                    
                    val_loss /= val_steps
                    
                    wandb.log({
                        "val_loss": val_loss,
                        "epoch": epoch + 1,
                        "step": step + 1,
                    })
                    print(f"\nStep {step+1} - Rank {rank} Train Loss: {train_loss/(step+1):.4f}, Validation Loss: {val_loss:.4f}")
                    
                    model.train()

            progress_bar.close()


            if (epoch + 1) % 4 == 0 and step == 0 and rank == 0:  # Check if at start of 4th, 8th, etc. epoch
                save_model_checkpoint(model, optimizer, scheduler, epoch, step, rank=rank)

            if rank == 0:  # Only rank 0 generates sample text
                START_CONTEXT = "As an AI language model,"
                token_ids = generate(
                    model=model.module,
                    device=device,
                    idx=text_to_token_ids(START_CONTEXT, tokenizer),
                    max_new_tokens=20,
                    context_len=args.context_len,
                )
                sample_text = token_ids_to_text(token_ids, tokenizer)
                print(f"\nSample text:", sample_text)
                
                wandb.log({
                    "sample_text": sample_text,
                    "epoch": epoch + 1,
                })

    except Exception as e:
        print(f"Rank {rank} encountered an error: {str(e)}")
        raise e
    finally:
        cleanup()

if __name__ == "__main__":
    args = parse_args()
    world_size = args.world_size
    mp.spawn(main, args=(args,), nprocs=world_size, join=True)