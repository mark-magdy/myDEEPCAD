import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import json
import matplotlib.pyplot as plt

from config import get_args
from models.deepcad import DeepCAD
from utils.data_utils import get_dataloader

def encode_dataset(model, dataloader, device):
    """Encode the entire dataset to latent space"""
    model.eval()
    all_latents = []
    all_filenames = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding dataset"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Encode to latent space
            latents = model.encode(batch['node_features'], batch['mask'])
            
            # Store latents and filenames
            all_latents.append(latents.cpu().numpy())
            all_filenames.extend(batch['file_names'])
    
    # Concatenate all latents
    all_latents = np.concatenate(all_latents, axis=0)
    
    return all_latents, all_filenames

def reconstruct_dataset(model, dataloader, device):
    """Reconstruct the entire dataset"""
    model.eval()
    all_reconstructions = []
    all_originals = []
    all_filenames = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Reconstructing dataset"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch['node_features'], batch['mask'], 0.0, None)
            
            # Process outputs
            op_types = torch.argmax(outputs['op_types'], dim=-1)
            params = outputs['params']
            sketch = outputs['sketch']
            
            # Store reconstructions, originals, and filenames
            for i in range(len(batch['file_names'])):
                # Find end of sequence (first zero or max_seq_len)
                seq_len = batch['op_mask'].size(1)
                for j in range(batch['op_mask'].size(1)):
                    if batch['op_mask'][i, j] == 0:
                        seq_len = j
                        break
                
                # Extract original sequence
                orig_op_types = batch['op_types'][i, :seq_len].cpu().numpy().tolist()
                orig_op_params = batch['op_params'][i, :seq_len].cpu().numpy().tolist()
                orig_sketch = batch['sketch_target'][i, :seq_len].cpu().numpy().tolist()
                orig_node_features = batch['node_features'][i].cpu().numpy().tolist()
                
                # Create original sample
                original = {
                    'op_types': orig_op_types,
                    'op_params': orig_op_params,
                    'sketch_target': orig_sketch,
                    'node_features': orig_node_features
                }
                
                # Extract reconstructed sequence
                recon_op_types = op_types[i, :seq_len].cpu().numpy().tolist()
                recon_op_params = params[i, :seq_len].cpu().numpy().tolist()
                recon_sketch = sketch[i, :seq_len].cpu().numpy().tolist()
                recon_node_features = outputs['output'][i].cpu().numpy().tolist()
                
                # Create reconstructed sample
                reconstruction = {
                    'op_types': recon_op_types,
                    'op_params': recon_op_params,
                    'sketch_target': recon_sketch,
                    'node_features': recon_node_features
                }
                
                all_originals.append(original)
                all_reconstructions.append(reconstruction)
                all_filenames.append(batch['file_names'][i])
    
    return all_originals, all_reconstructions, all_filenames

def decode_latents(model, latents, device):
    """Decode latent vectors to CAD models"""
    model.eval()
    all_samples = []
    
    with torch.no_grad():
        # Process in batches
        batch_size = 32
        num_batches = (latents.shape[0] + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Decoding latents"):
            # Get batch
            batch_latents = latents[i * batch_size:(i + 1) * batch_size]
            batch_latents_tensor = torch.tensor(batch_latents, dtype=torch.float32, device=device)
            
            # Decode
            outputs = model.decode(batch_latents_tensor)
            
            # Process outputs
            op_types = torch.argmax(outputs['op_types'], dim=-1)
            params = outputs['params']
            sketch = outputs['sketch']
            
            # Convert to list of samples
            for j in range(len(batch_latents)):
                # Find end of sequence (first zero or max_seq_len)
                seq_len = op_types.size(1)
                for k in range(op_types.size(1)):
                    if op_types[j, k] == 0:
                        seq_len = k + 1
                        break
                
                # Extract sequence
                sample_op_types = op_types[j, :seq_len].cpu().numpy().tolist()
                sample_params = params[j, :seq_len].cpu().numpy().tolist()
                sample_sketch = sketch[j, :seq_len].cpu().numpy().tolist()
                
                # Create sample
                sample = {
                    'op_types': sample_op_types,
                    'op_params': sample_params,
                    'sketch_target': sample_sketch
                }
                
                all_samples.append(sample)
    
    return all_samples

def main(args):
    """Main testing function"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create model
    model = DeepCAD(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    # Move model to device
    device = torch.device(args.device)
    model = model.to(device)
    
    # Load checkpoint
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            print(f"No checkpoint found at {args.checkpoint}")
            return
    
    # Create dataloader
    dataloader = get_dataloader(
        args.data_dir,
        args.split,
        batch_size=args.batch_size,
        shuffle=False,
        max_seq_len=args.max_seq_len
    )
    
    # Run the requested mode
    if args.mode == 'encode':
        print(f"Encoding {args.split} dataset to latent space...")
        latents, filenames = encode_dataset(model, dataloader, device)
        
        # Save latents
        np.save(os.path.join(output_dir, f"{args.split}_latents.npy"), latents)
        with open(os.path.join(output_dir, f"{args.split}_filenames.json"), 'w') as f:
            json.dump(filenames, f)
        
        print(f"Encoded {len(filenames)} samples to latent space")
        print(f"Latent shape: {latents.shape}")
        
    elif args.mode == 'reconstruct':
        print(f"Reconstructing {args.split} dataset...")
        originals, reconstructions, filenames = reconstruct_dataset(model, dataloader, device)
        
        # Save reconstructions
        for i, (original, reconstruction, filename) in enumerate(zip(originals, reconstructions, filenames)):
            # Save to separate files
            with open(os.path.join(output_dir, f"original_{i}.json"), 'w') as f:
                json.dump(original, f)
            
            with open(os.path.join(output_dir, f"reconstruction_{i}.json"), 'w') as f:
                json.dump(reconstruction, f)
        
        print(f"Reconstructed {len(filenames)} samples")
        
    elif args.mode == 'decode':
        if args.latent_file:
            print(f"Decoding latents from {args.latent_file}...")
            latents = np.load(args.latent_file)
            samples = decode_latents(model, latents, device)
            
            # Save samples
            for i, sample in enumerate(samples):
                with open(os.path.join(output_dir, f"decoded_{i}.json"), 'w') as f:
                    json.dump(sample, f)
            
            print(f"Decoded {len(samples)} samples")
        else:
            print("No latent file specified")
    
    elif args.mode == 'generate':
        print(f"Generating {args.num_samples} samples...")
        # Sample from prior
        latents = np.random.randn(args.num_samples, args.latent_dim)
        samples = decode_latents(model, latents, device)
        
        # Save samples
        for i, sample in enumerate(samples):
            with open(os.path.join(output_dir, f"generated_{i}.json"), 'w') as f:
                json.dump(sample, f)
        
        print(f"Generated {len(samples)} samples")
    
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="DeepCAD Testing")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/deepcad', help='Data directory')
    parser.add_argument('--max_seq_len', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=3, help='Input dimension for node features')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Testing parameters
    parser.add_argument('--mode', type=str, required=True, choices=['encode', 'reconstruct', 'decode', 'generate'], help='Test mode')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory')
    parser.add_argument('--latent_file', type=str, default=None, help='Path to latent file for decode mode')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run test
    main(args) 