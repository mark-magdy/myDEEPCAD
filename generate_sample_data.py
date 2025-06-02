import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm

def generate_random_cad_sample(max_nodes=50, max_ops=20, node_dim=3, param_dim=10, sketch_dim=128):
    """Generate a random CAD sample"""
    # Random number of nodes
    n_nodes = random.randint(5, max_nodes)
    
    # Node features: random 3D points
    node_features = np.random.randn(n_nodes, node_dim).tolist()
    
    # Adjacency matrix: random sparse graph
    adjacency = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        # Connect to random nodes
        n_connections = random.randint(1, min(5, n_nodes - 1))
        connections = random.sample(range(n_nodes), n_connections)
        for j in connections:
            if i != j:
                adjacency[i, j] = 1.0
    
    # Create mask
    mask = np.ones(n_nodes).tolist()
    
    # Random number of operations
    n_ops = random.randint(1, max_ops)
    
    # Operation types: integers between 0 and 9
    op_types = [random.randint(0, 9) for _ in range(n_ops)]
    
    # Operation parameters
    op_params = np.random.randn(n_ops, param_dim).tolist()
    
    # Operation mask
    op_mask = np.ones(n_ops).tolist()
    
    # Sketch target
    sketch_target = np.random.randn(n_ops, sketch_dim).tolist()
    
    return {
        'node_features': node_features,
        'adjacency': adjacency.tolist(),
        'mask': mask,
        'op_types': op_types,
        'op_params': op_params,
        'op_mask': op_mask,
        'sketch_target': sketch_target
    }

def generate_dataset(output_dir, num_samples=100, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Generate a dataset of random CAD samples"""
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Calculate number of samples for each split
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    # Generate training samples
    for i in tqdm(range(num_train), desc="Generating training samples"):
        sample = generate_random_cad_sample()
        with open(os.path.join(train_dir, f'sample_{i:05d}.json'), 'w') as f:
            json.dump(sample, f)
    
    # Generate validation samples
    for i in tqdm(range(num_val), desc="Generating validation samples"):
        sample = generate_random_cad_sample()
        with open(os.path.join(val_dir, f'sample_{i:05d}.json'), 'w') as f:
            json.dump(sample, f)
    
    # Generate test samples
    for i in tqdm(range(num_test), desc="Generating test samples"):
        sample = generate_random_cad_sample()
        with open(os.path.join(test_dir, f'sample_{i:05d}.json'), 'w') as f:
            json.dump(sample, f)
    
    print(f"Generated {num_samples} samples:")
    print(f"- Training samples: {num_train}")
    print(f"- Validation samples: {num_val}")
    print(f"- Test samples: {num_test}")

def validate_samples(data_dir):
    """Validate that all samples can be loaded correctly"""
    print("Validating samples...")
    
    from utils.data_utils import CADDataset
    
    # Try loading training data
    train_dataset = CADDataset(data_dir, 'train')
    for i in range(len(train_dataset)):
        try:
            sample = train_dataset[i]
        except Exception as e:
            print(f"Error loading training sample {i}: {e}")
    
    # Try loading validation data
    val_dataset = CADDataset(data_dir, 'val')
    for i in range(len(val_dataset)):
        try:
            sample = val_dataset[i]
        except Exception as e:
            print(f"Error loading validation sample {i}: {e}")
    
    # Try loading test data
    test_dataset = CADDataset(data_dir, 'test')
    for i in range(len(test_dataset)):
        try:
            sample = test_dataset[i]
        except Exception as e:
            print(f"Error loading test sample {i}: {e}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate dataset
    generate_dataset('data/deepcad', num_samples=100, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    
    # Validate samples
    validate_samples('data/deepcad') 