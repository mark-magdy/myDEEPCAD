import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def create_padding_mask(lengths, max_len=None):
    """
    Create padding mask for transformer models
    Args:
        lengths: list of sequence lengths
        max_len: maximum sequence length
    Returns:
        mask: [batch_size, max_len] tensor, 1 for valid positions, 0 for padding
    """
    if max_len is None:
        max_len = max(lengths)
    
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    
    return mask

class CADDataset(Dataset):
    """Dataset for CAD models"""
    def __init__(self, data_dir, split='train', max_seq_len=100):
        self.data_dir = os.path.join(data_dir, split)
        self.max_seq_len = max_seq_len
        
        # Check if directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            
        # Get list of data files
        self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
        if len(self.data_files) == 0:
            print(f"Warning: No data files found in {self.data_dir}")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load data from JSON file
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            # Return a default empty sample
            return self.get_default_sample()
        
        # Convert data to tensors
        try:
            # Node features
            node_features = torch.tensor(data.get('node_features', []), dtype=torch.float32)
            
            # Ensure node_features has at least one element
            if node_features.numel() == 0:
                node_features = torch.zeros((1, 3), dtype=torch.float32)
            
            # Truncate if too long
            if node_features.size(0) > self.max_seq_len:
                node_features = node_features[:self.max_seq_len]
            
            # Adjacency matrix
            adjacency = torch.tensor(data.get('adjacency', []), dtype=torch.float32)
            if adjacency.numel() == 0:
                adjacency = torch.zeros((node_features.size(0), node_features.size(0)), dtype=torch.float32)
            
            # Truncate if too long
            if adjacency.size(0) > self.max_seq_len:
                adjacency = adjacency[:self.max_seq_len, :self.max_seq_len]
            
            # Create mask
            mask = torch.ones(node_features.size(0), dtype=torch.float32)
            
            # Operation types and parameters
            op_types = torch.tensor(data.get('op_types', []), dtype=torch.long)
            if op_types.numel() == 0:
                op_types = torch.zeros((1,), dtype=torch.long)
            
            # Truncate if too long
            if op_types.size(0) > self.max_seq_len:
                op_types = op_types[:self.max_seq_len]
            
            op_params = torch.tensor(data.get('op_params', []), dtype=torch.float32)
            if op_params.numel() == 0:
                op_params = torch.zeros((op_types.size(0), 10), dtype=torch.float32)
            elif op_params.dim() == 1:
                op_params = op_params.unsqueeze(0)
            
            # Truncate if too long
            if op_params.size(0) > self.max_seq_len:
                op_params = op_params[:self.max_seq_len]
            
            # Operation mask
            op_mask = torch.ones(op_types.size(0), dtype=torch.float32)
            
            # Sketch target
            sketch_target = torch.tensor(data.get('sketch_target', []), dtype=torch.float32)
            if sketch_target.numel() == 0:
                sketch_target = torch.zeros((op_types.size(0), 128), dtype=torch.float32)
            elif sketch_target.dim() == 1:
                sketch_target = sketch_target.unsqueeze(0)
            
            # Truncate if too long
            if sketch_target.size(0) > self.max_seq_len:
                sketch_target = sketch_target[:self.max_seq_len]
            
            return {
                'node_features': node_features,
                'adjacency': adjacency,
                'mask': mask,
                'op_types': op_types,
                'op_params': op_params,
                'op_mask': op_mask,
                'sketch_target': sketch_target,
                'file_name': self.data_files[idx]
            }
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return self.get_default_sample()
    
    def get_default_sample(self):
        """Return a default empty sample"""
        node_features = torch.zeros((1, 3), dtype=torch.float32)
        adjacency = torch.zeros((1, 1), dtype=torch.float32)
        mask = torch.ones(1, dtype=torch.float32)
        op_types = torch.zeros((1,), dtype=torch.long)
        op_params = torch.zeros((1, 10), dtype=torch.float32)
        op_mask = torch.ones(1, dtype=torch.float32)
        sketch_target = torch.zeros((1, 128), dtype=torch.float32)
        
        return {
            'node_features': node_features,
            'adjacency': adjacency,
            'mask': mask,
            'op_types': op_types,
            'op_params': op_params,
            'op_mask': op_mask,
            'sketch_target': sketch_target,
            'file_name': 'default.json'
        }

def collate_cad_batch(batch):
    """
    Custom collate function for CAD data
    Args:
        batch: list of samples from CADDataset
    Returns:
        batch_dict: dictionary of batched tensors
    """
    # Get batch size
    batch_size = len(batch)
    
    # Get maximum sequence length in this batch
    max_node_len = max([sample['node_features'].size(0) for sample in batch])
    max_op_len = max([sample['op_types'].size(0) for sample in batch])
    
    # Initialize tensors
    node_features = torch.zeros(batch_size, max_node_len, batch[0]['node_features'].size(1))
    adjacency = torch.zeros(batch_size, max_node_len, max_node_len)
    mask = torch.zeros(batch_size, max_node_len)
    op_types = torch.zeros(batch_size, max_op_len, dtype=torch.long)
    op_params = torch.zeros(batch_size, max_op_len, batch[0]['op_params'].size(1))
    op_mask = torch.zeros(batch_size, max_op_len)
    sketch_target = torch.zeros(batch_size, max_op_len, batch[0]['sketch_target'].size(1))
    
    file_names = []
    
    # Fill tensors
    for i, sample in enumerate(batch):
        # Node features
        n_nodes = sample['node_features'].size(0)
        node_features[i, :n_nodes] = sample['node_features']
        
        # Adjacency matrix
        adjacency[i, :n_nodes, :n_nodes] = sample['adjacency']
        
        # Mask
        mask[i, :n_nodes] = sample['mask']
        
        # Operation types
        n_ops = sample['op_types'].size(0)
        op_types[i, :n_ops] = sample['op_types']
        
        # Operation parameters
        op_params[i, :n_ops] = sample['op_params']
        
        # Operation mask
        op_mask[i, :n_ops] = sample['op_mask']
        
        # Sketch target
        sketch_target[i, :n_ops] = sample['sketch_target']
        
        # File name
        file_names.append(sample['file_name'])
    
    return {
        'node_features': node_features,
        'adjacency': adjacency,
        'mask': mask,
        'op_types': op_types,
        'op_params': op_params,
        'op_mask': op_mask,
        'sketch_target': sketch_target,
        'file_names': file_names
    }

def get_dataloader(data_dir, split='train', batch_size=32, shuffle=True, max_seq_len=100):
    """Create a dataloader for the CAD dataset"""
    dataset = CADDataset(data_dir, split, max_seq_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        collate_fn=collate_cad_batch,
        pin_memory=False,  # Set to False for Windows compatibility
        drop_last=False
    )
    
    return dataloader 