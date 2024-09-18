import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np 
import random 

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def tokenize(tokenizer,data):
    encoded_data = []
    
    # Use tqdm for progress bar
    for item in tqdm(data, desc="Tokenizing"):
        encoded_text = tokenizer.encode(item['text'])
        encoded_data.append(encoded_text)
    
    # Flatten the list of encoded texts into a single long sequence
    return [token for text in encoded_data for token in text]


class GPTDatasetV1(Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.token_ids = token_ids
        self.max_length = max_length
        self.stride = stride
        self.num_samples = (len(token_ids) - max_length) // stride + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.max_length
        
        input_chunk = self.token_ids[start_idx:end_idx]
        target_chunk = self.token_ids[start_idx+1:end_idx+1]
        
        return torch.tensor(input_chunk), torch.tensor(target_chunk)