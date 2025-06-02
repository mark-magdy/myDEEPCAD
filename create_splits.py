import os
import json
import shutil
import random
from pathlib import Path

def create_data_splits(source_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Split data into train and validation sets
    
    Args:
        source_dir (str): Directory containing all JSON files
        output_dir (str): Directory to save split data
        train_ratio (float): Ratio of training data
        seed (int): Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    random.shuffle(json_files)
    
    # Split files
    split_idx = int(len(json_files) * train_ratio)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]
    
    # Copy files to respective directories
    for file in train_files:
        shutil.copy2(
            os.path.join(source_dir, file),
            os.path.join(train_dir, file)
        )
    
    for file in val_files:
        shutil.copy2(
            os.path.join(source_dir, file),
            os.path.join(val_dir, file)
        )
    
    print(f"Created train set with {len(train_files)} files")
    print(f"Created validation set with {len(val_files)} files")

def validate_json_format(file_path):
    """
    Validate JSON file format
    
    Args:
        file_path (str): Path to JSON file
    
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = [
        'node_features',
        'adjacency',
        'mask',
        'op_types',
        'op_params',
        'op_mask',
        'sketch_target'
    ]
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check all required keys exist
        for key in required_keys:
            if key not in data:
                print(f"Missing key: {key} in {file_path}")
                return False
        
        # Validate data types and shapes
        if not isinstance(data['node_features'], list):
            print(f"node_features should be a list in {file_path}")
            return False
        
        if not isinstance(data['adjacency'], list):
            print(f"adjacency should be a list in {file_path}")
            return False
        
        return True
    
    except json.JSONDecodeError:
        print(f"Invalid JSON format in {file_path}")
        return False
    except Exception as e:
        print(f"Error validating {file_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create train/val splits for DeepCAD dataset')
    parser.add_argument('--source_dir', type=str, required=True,
                      help='Directory containing all JSON files')
    parser.add_argument('--output_dir', type=str, default='data/deepcad',
                      help='Directory to save split data')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                      help='Ratio of training data')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate source directory
    if not os.path.exists(args.source_dir):
        print(f"Source directory {args.source_dir} does not exist!")
        return
    
    # Create splits
    create_data_splits(
        args.source_dir,
        args.output_dir,
        args.train_ratio,
        args.seed
    )
    
    # Validate all files
    print("\nValidating JSON files...")
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.json')]
    val_files = [f for f in os.listdir(val_dir) if f.endswith('.json')]
    
    all_valid = True
    for file in train_files + val_files:
        dir_path = train_dir if file in train_files else val_dir
        if not validate_json_format(os.path.join(dir_path, file)):
            all_valid = False
    
    if all_valid:
        print("All JSON files are valid!")
    else:
        print("Some JSON files are invalid. Please check the errors above.")

if __name__ == "__main__":
    import argparse
    main()
