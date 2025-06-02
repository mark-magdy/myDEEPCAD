# data_utils.py
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def pad_tensor(tensor_list, pad_value=0):
    """
    Pad a list of tensors to the maximum shape in each dimension.
    Args:
        tensor_list: list of torch.Tensor, each of shape [d1, d2, ...]
        pad_value: value to use for padding
    Returns:
        torch.Tensor of shape [batch, max_d1, max_d2, ...]
    """
    # Find max shape in each dimension
    max_shape = list(tensor_list[0].shape)
    for tensor in tensor_list[1:]:
        for i, dim in enumerate(tensor.shape):
            if dim > max_shape[i]:
                max_shape[i] = dim
    
    # Pad each tensor
    padded = []
    for tensor in tensor_list:
        pad_sizes = []
        for i, dim in enumerate(tensor.shape):
            pad_sizes.append(max_shape[i] - dim)
        # torch.nn.functional.pad expects reverse order and pairs
        pad = []
        for p in reversed(pad_sizes):
            pad.extend([0, p])
        padded_tensor = torch.nn.functional.pad(tensor, pad, value=pad_value)
        padded.append(padded_tensor)
    return torch.stack(padded)

def collate_fn(batch):
    """Custom collate function to handle variable-sized inputs and pad all dims."""
    # Extract all fields
    node_features = [item['node_features'] for item in batch]
    adjacency = [item['adjacency'] for item in batch]
    mask = [item['mask'] for item in batch]
    op_types = [item['op_types'] for item in batch]
    op_params = [item['op_params'] for item in batch]
    op_mask = [item['op_mask'] for item in batch]
    sketch_target = [item['sketch_target'] for item in batch]
    
    # Pad tensors
    node_features = pad_tensor(node_features)
    adjacency = pad_tensor(adjacency)
    mask = pad_tensor(mask)
    op_types = pad_tensor(op_types, pad_value=-1)  # Use -1 for padding in op_types
    op_params = pad_tensor(op_params)
    op_mask = pad_tensor(op_mask)
    sketch_target = pad_tensor(sketch_target)
    
    # Ensure masks are binary
    mask = (mask > 0).float()
    op_mask = (op_mask > 0).float()
    
    return {
        'node_features': node_features,
        'adjacency': adjacency,
        'mask': mask,
        'op_types': op_types,
        'op_params': op_params,
        'op_mask': op_mask,
        'sketch_target': sketch_target
    }

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
        
        # Convert lists to tensors and ensure correct types
        node_features = torch.tensor(data['node_features'], dtype=torch.float32)
        adjacency = torch.tensor(data['adjacency'], dtype=torch.float32)
        mask = torch.tensor(data['mask'], dtype=torch.float32)
        op_types = torch.tensor(data['op_types'], dtype=torch.long)
        op_params = torch.tensor(data['op_params'], dtype=torch.float32)
        op_mask = torch.tensor(data['op_mask'], dtype=torch.float32)
        sketch_target = torch.tensor(data['sketch_target'], dtype=torch.float32)
        
        # Ensure masks are binary
        mask = (mask > 0).float()
        op_mask = (op_mask > 0).float()
        
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
        pin_memory=True,
        collate_fn=collate_fn  # Use custom collate function
    )
    
    return dataloader