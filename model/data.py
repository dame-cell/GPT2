import numpy as np
import torch
from tqdm import tqdm
from utils import tokenize 
import tiktoken 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--sample_size', type=int, default=300_000, help="sample size for data")
    parser.add_argument('--data_name', type=str, default="eliplutchok/fineweb-small-sample", help="data to be used in processing and training")
    return parser.parse_args()

def preprocess_and_save(token_ids, max_length, stride, save_path):
    inputs = []
    targets = []
    
    # Calculate the number of steps for tqdm
    num_steps = (len(token_ids) - max_length) // stride + 1
    
    # Preprocess token_ids with tqdm to show progress
    for i in tqdm(range(num_steps), desc=f"Processing {save_path}"):
        start_idx = i * stride
        input_chunk = token_ids[start_idx:start_idx + max_length]
        target_chunk = token_ids[start_idx + 1:start_idx + max_length + 1]
        inputs.append(input_chunk)
        targets.append(target_chunk)
    
    np.savez(save_path, inputs=np.array(inputs), targets=np.array(targets))

def split_and_save(token_ids, split_ratio=0.9, max_length=512, stride=256):
    tokens_tensor = torch.tensor(token_ids)
    num_samples = len(tokens_tensor)

    split_idx = int(num_samples * split_ratio)
    train_tokens = tokens_tensor[:split_idx]
    test_tokens = tokens_tensor[split_idx:]

    # Preprocess and save train and test data
    preprocess_and_save(train_tokens, max_length, stride, "train_data.npz")
    preprocess_and_save(test_tokens, max_length, stride, "test_data.npz")

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    args = parse_args()
    from datasets import load_dataset 

    ds = load_dataset(args.data_name, split='train')
    ds = ds.select(range(min(args.sample_size, len(ds))))
    tokens = tokenize(tokenizer,ds)
    split_and_save(tokens)
