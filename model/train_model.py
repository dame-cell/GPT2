import torch
import torch.nn as nn 
import tiktoken 
import argparse
import wandb 
from tqdm.auto import tqdm 
from datasets import load_dataset
import torch.nn.functional  as F 
from modeling_gpt2 import GPT2
from utils import tokenize ,GPTDatasetV1, setup_seed, save_model_checkpoint
from torch.utils.data import DataLoader
from generate import text_to_token_ids ,  token_ids_to_text , generate 
from transformers import get_linear_schedule_with_warmup



def parse_args():
    parser = argparse.ArgumentParser(description="Train gpt2 on sample-fineweb")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs (default: 2)")
    parser.add_argument('--train_data', type=str, help="path to the train npz file")
    parser.add_argument('--test_data', type=str,  help="path to the test npz file")
    parser.add_argument('--vocab_size', type=int, default=50257, help="the tokenizer vocab size")
    parser.add_argument('--emb_dim', type=int, default=684, help="the embedding dimension of the model")
    parser.add_argument('--num_layers', type=int, default=6, help="the number of layers for the transformers")
    parser.add_argument('--num_heads', type=int, default=12, help="the number of attentions heads for the transformers")
    parser.add_argument('--tokenizer_name', type=str,default="gpt2", help="tokenizer name that will be used by tiktoken")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and validation (default: 128)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for optimizer (default: 1e-4)")
    parser.add_argument('--eval_interval', type=int, default=2, help="Evaluation interval during training (default: 100 steps)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument('--max_length', type=int, default=512,help="max lenght for our model ")
    parser.add_argument('--context_len', type=int, default=20,help="max length for the model generatng text")
    parser.add_argument('--data_name',type=str,default="eliplutchok/fineweb-small-sample",help="dataset to used for training but from huggingface")
    parser.add_argument('--sample_size',type=int,default="300000",help="How many rows to train the model on (max is 700k)")
    parser.add_argument('--stride', type=int, default=256, help="The stride defines how much overlap there is between the consecutive sequences.")

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Initialize Weights & Biases
    wandb.login(key="04098c64a0b88d5f4ff90335b7f75613041420c6")
    wandb.init(
        project="gpt2-sample-fineweb",
        config=args
    )
    
    tokenizer = tiktoken.get_encoding(args.tokenizer_name)


    START_CONTEXT = "As an AI language model,"


    train_dataset = GPTDatasetV1(args.train_data)
    val_dataset = GPTDatasetV1(args.test_data)



    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)



    model = GPT2(args=args)
    model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)  
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)    
    scaler = torch.GradScaler()

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0  # To accumulate train loss for the epoch
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, (input_batch, target_batch) in progress_bar:
            input_batch, target_batch = input_batch.to(DEVICE), target_batch.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_batch)
                loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            
            # Backward pass and optimization step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()  # Accumulate train loss
            scheduler.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
                "step": step + 1
            })
            if (step + 1) % 2000 == 0:
                if rank == 0:
                    save_model_checkpoint(model, optimizer, scheduler, epoch, step,rank=rank)

            if (step + 1) % args.eval_interval == 0:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                
                # Validation Loop
                val_progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validating")
                for val_step, (val_input_batch, val_target_batch) in val_progress_bar:
                    val_input_batch, val_target_batch = val_input_batch.to(DEVICE), val_target_batch.to(DEVICE)
                    
                    with torch.no_grad():
                        val_logits = model(val_input_batch)
                        val_loss += F.cross_entropy(val_logits.flatten(0, 1), val_target_batch.flatten()).item()
                    
                    val_steps += 1
                
                val_loss /= val_steps  # Average validation loss
                
                # Log validation loss to wandb
                wandb.log({
                    "val_loss": val_loss,
                    "epoch": epoch + 1,
                    "step": step + 1
                })

                print(f"\nStep {step+1} - Train Loss: {train_loss/(step+1):.4f}, Validation Loss: {val_loss:.4f}")
                model.train()  # Switch back to train mode after validation


        # Generate text at the end of each epoch
        token_ids = generate(
            model=model,
            device=DEVICE,
            idx=text_to_token_ids(START_CONTEXT, tokenizer),
            max_new_tokens=20,
            context_len=args.context_len,
        )
        sample_text = token_ids_to_text(token_ids, tokenizer)
        print("Sample text:", sample_text)
        
        # Log generated sample text to wandb
        wandb.log({
            "sample_text": sample_text,
            "epoch": epoch + 1
        })

    # Finish the wandb run
    wandb.finish()