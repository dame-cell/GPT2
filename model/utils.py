import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np 
import random 
import os 

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
    
    return [token for text in encoded_data for token in text]

def save_model_checkpoint(model, optimizer, scheduler, epoch, step, rank, save_dir="checkpoints"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"gpt2_epoch{epoch+1}_step{step+1}_gpu{rank}.pt")
    torch.save({
        'epoch': epoch + 1,
        'step': step + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_path)
    print(f"Model saved at {save_path}")

class GPTDatasetV1(Dataset):
    def __init__(self, npz_file_path):
        # Load precomputed inputs and targets from .npz file
        data = np.load(npz_file_path)
        self.inputs = data['inputs']
        self.targets = data['targets']
        
        self.num_samples = len(self.inputs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_chunk = self.inputs[idx]
        target_chunk = self.targets[idx]
        return torch.tensor(input_chunk), torch.tensor(target_chunk)

