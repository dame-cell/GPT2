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
        self.input_ids = []
        self.target_ids = []

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            # No need to truncate as we're already slicing to max_length

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]