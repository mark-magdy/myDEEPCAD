import os
import json
import numpy as np
from tqdm import tqdm

def create_sample_data(num_samples=100, max_nodes=50, max_ops=20):
    """Generate sample data for training."""
    for i in tqdm(range(num_samples)):
        # Random number of nodes for this sample
        num_nodes = np.random.randint(10, max_nodes)
        
        # Generate node features [num_nodes, 3]
        node_features = np.random.randn(num_nodes, 3).tolist()
        
        # Generate adjacency matrix [num_nodes, num_nodes]
        adjacency = np.random.randint(0, 2, (num_nodes, num_nodes)).tolist()
        
        # Generate mask [num_nodes]
        mask = np.ones(num_nodes).tolist()
        
        # Generate operation data
        op_types = np.random.randint(0, 5, max_ops).tolist()  # 5 different operation types
        op_params = np.random.randn(max_ops, 10).tolist()  # 10 parameters per operation
        op_mask = np.ones(max_ops).tolist()
        
        # Generate sketch target [max_ops, 128]
        sketch_target = np.random.randn(max_ops, 128).tolist()
        
        sample = {
            'node_features': node_features,
            'adjacency': adjacency,
            'mask': mask,
            'op_types': op_types,
            'op_params': op_params,
            'op_mask': op_mask,
            'sketch_target': sketch_target
        }
        
        # Save each sample as a separate file
        save_data(sample, f'data/deepcad/train/sample_{i}.json')
    
    print(f"Generated {num_samples} training samples")

def save_data(data, filename):
    """Save data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f)

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data/deepcad', exist_ok=True)
    os.makedirs('data/deepcad/train', exist_ok=True)
    os.makedirs('data/deepcad/val', exist_ok=True)
    os.makedirs('data/deepcad/test', exist_ok=True)
    
    # Generate data for each split
    create_sample_data(num_samples=100)  # Training data
    create_sample_data(num_samples=20)   # Validation data
    create_sample_data(num_samples=20)   # Test data
    
    print("Sample data generated successfully!")

if __name__ == '__main__':
    main() 