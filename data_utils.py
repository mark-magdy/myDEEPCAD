# data_utils.py
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

class CADDataset(Dataset):
    """Dataset for CAD models"""
    def __init__(self, data_dir, split='train'):
        self.data_dir = os.path.join(data_dir, split)
        self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load data from JSON file
        with open(os.path.join(self.data_dir, self.data_files[idx]), 'r') as f:
            data = json.load(f)
        
        # Convert lists to tensors
        node_features = torch.tensor(data['node_features'], dtype=torch.float32)
        adjacency = torch.tensor(data['adjacency'], dtype=torch.float32)
        mask = torch.tensor(data['mask'], dtype=torch.float32)
        op_types = torch.tensor(data['op_types'], dtype=torch.long)
        op_params = torch.tensor(data['op_params'], dtype=torch.float32)
        op_mask = torch.tensor(data['op_mask'], dtype=torch.float32)
        sketch_target = torch.tensor(data['sketch_target'], dtype=torch.float32)
        
        return {
            'node_features': node_features,
            'adjacency': adjacency,
            'mask': mask,
            'op_types': op_types,
            'op_params': op_params,
            'op_mask': op_mask,
            'sketch_target': sketch_target
        }

def get_dataloader(data_dir, split='train', batch_size=32, shuffle=True):
    """Create a dataloader for the CAD dataset"""
    dataset = CADDataset(data_dir, split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    return dataloader