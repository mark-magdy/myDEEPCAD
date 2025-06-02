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

def compute_reconstruction_accuracy(original, reconstructed, threshold=0.1):
    """Compute reconstruction accuracy"""
    # Handle different sequence lengths
    if original.size(1) != reconstructed.size(1):
        if original.size(1) > reconstructed.size(1):
            # Truncate original to match reconstructed
            original = original[:, :reconstructed.size(1), :]
        else:
            # Truncate reconstructed to match original
            reconstructed = reconstructed[:, :original.size(1), :]
    
    diff = torch.abs(original - reconstructed)
    accuracy = (diff < threshold).float().mean()
    return accuracy.item()

def compute_operation_accuracy(original, reconstructed):
    """Compute operation type accuracy"""
    # Handle different sequence lengths
    if original.size(1) != reconstructed.size(1):
        if original.size(1) > reconstructed.size(1):
            # Truncate original to match reconstructed
            original = original[:, :reconstructed.size(1)]
        else:
            # Truncate reconstructed to match original
            reconstructed = reconstructed[:, :original.size(1)]
    
    accuracy = (original == reconstructed).float().mean()
    return accuracy.item()

def evaluate(model, dataloader, device, args):
    """Evaluate the model"""
    model.eval()
    
    # Metrics
    recon_accuracies = []
    op_accuracies = []
    param_errors = []
    sketch_errors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch['node_features'], batch['mask'], 0.0, None)
            
            # Compute metrics
            # Reconstruction accuracy
            recon_acc = compute_reconstruction_accuracy(
                batch['node_features'], 
                outputs['output'],
                threshold=0.1
            )
            recon_accuracies.append(recon_acc)
            
            # Operation type accuracy
            op_pred = torch.argmax(outputs['op_types'], dim=-1)
            op_acc = compute_operation_accuracy(batch['op_types'], op_pred)
            op_accuracies.append(op_acc)
            
            # Parameter error
            # Handle different sequence lengths
            params = outputs['params']
            op_params = batch['op_params']
            if params.size(1) != op_params.size(1):
                if params.size(1) > op_params.size(1):
                    # Truncate params to match op_params
                    params = params[:, :op_params.size(1), :]
                else:
                    # Truncate op_params to match params
                    op_params = op_params[:, :params.size(1), :]
            param_error = torch.mean(torch.abs(params - op_params))
            param_errors.append(param_error.item())
            
            # Sketch error
            # Handle different sequence lengths
            sketch = outputs['sketch']
            sketch_target = batch['sketch_target']
            if sketch.size(1) != sketch_target.size(1):
                if sketch.size(1) > sketch_target.size(1):
                    # Truncate sketch to match sketch_target
                    sketch = sketch[:, :sketch_target.size(1), :]
                else:
                    # Truncate sketch_target to match sketch
                    sketch_target = sketch_target[:, :sketch.size(1), :]
            sketch_error = torch.mean(torch.abs(sketch - sketch_target))
            sketch_errors.append(sketch_error.item())
    
    # Compute average metrics
    avg_recon_acc = np.mean(recon_accuracies)
    avg_op_acc = np.mean(op_accuracies)
    avg_param_error = np.mean(param_errors)
    avg_sketch_error = np.mean(sketch_errors)
    
    return {
        'recon_accuracy': avg_recon_acc,
        'op_accuracy': avg_op_acc,
        'param_error': avg_param_error,
        'sketch_error': avg_sketch_error
    }

def generate_samples(model, num_samples, device, max_seq_len=100):
    """Generate samples from the model"""
    model.eval()
    
    with torch.no_grad():
        # Sample from prior
        z = torch.randn(num_samples, model.encoder.fc_mu.out_features).to(device)
        
        # Decode
        outputs = model.decode(z)
        
        # Process outputs
        op_types = torch.argmax(outputs['op_types'], dim=-1)
        params = outputs['params']
        sketch = outputs['sketch']
        
        # Convert to list of samples
        samples = []
        for i in range(num_samples):
            # Find end of sequence (first zero or max_seq_len)
            seq_len = max_seq_len
            for j in range(max_seq_len):
                if op_types[i, j] == 0:
                    seq_len = j + 1
                    break
            
            # Extract sequence
            sample_op_types = op_types[i, :seq_len].cpu().numpy().tolist()
            sample_params = params[i, :seq_len].cpu().numpy().tolist()
            sample_sketch = sketch[i, :seq_len].cpu().numpy().tolist()
            
            # Create sample
            sample = {
                'op_types': sample_op_types,
                'op_params': sample_params,
                'sketch_target': sample_sketch
            }
            
            samples.append(sample)
    
    return samples

def visualize_reconstruction(original, reconstructed, output_dir):
    """Visualize original and reconstructed samples"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize a few samples
    for i in range(min(5, len(original))):
        # Get original and reconstructed data
        orig_data = original[i]
        recon_data = reconstructed[i]
        
        # Plot node features
        plt.figure(figsize=(12, 6))
        
        # Original
        plt.subplot(1, 2, 1)
        plt.scatter(orig_data['node_features'][:, 0], orig_data['node_features'][:, 1], c=orig_data['node_features'][:, 2])
        plt.title(f"Original Sample {i+1}")
        plt.colorbar()
        
        # Reconstructed
        plt.subplot(1, 2, 2)
        plt.scatter(recon_data['node_features'][:, 0], recon_data['node_features'][:, 1], c=recon_data['node_features'][:, 2])
        plt.title(f"Reconstructed Sample {i+1}")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"reconstruction_{i+1}.png"))
        plt.close()
        
        # Save data
        with open(os.path.join(output_dir, f"original_{i+1}.json"), 'w') as f:
            json.dump(orig_data, f, indent=2)
        
        with open(os.path.join(output_dir, f"reconstructed_{i+1}.json"), 'w') as f:
            json.dump(recon_data, f, indent=2)

def main(args):
    """Main evaluation function"""
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
    test_loader = get_dataloader(
        args.data_dir,
        'test',
        batch_size=args.batch_size,
        shuffle=False,
        max_seq_len=args.max_seq_len
    )
    
    # Evaluate model
    metrics = evaluate(model, test_loader, device, args)
    
    # Print metrics
    print("Evaluation metrics:")
    print(f"  Reconstruction accuracy: {metrics['recon_accuracy']:.4f}")
    print(f"  Operation accuracy: {metrics['op_accuracy']:.4f}")
    print(f"  Parameter error: {metrics['param_error']:.4f}")
    print(f"  Sketch error: {metrics['sketch_error']:.4f}")
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate samples
    if args.generate_samples:
        print(f"Generating {args.num_samples} samples...")
        samples = generate_samples(model, args.num_samples, device, args.max_seq_len)
        
        # Save samples
        for i, sample in enumerate(samples):
            with open(os.path.join(output_dir, f"generated_{i+1}.json"), 'w') as f:
                json.dump(sample, f, indent=2)
        
        print(f"Generated {len(samples)} samples")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="DeepCAD Evaluation")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/deepcad', help='Data directory')
    parser.add_argument('--max_seq_len', type=int, default=100, help='Maximum sequence length')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=3, help='Input dimension for node features')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--generate_samples', action='store_true', help='Generate samples')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run evaluation
    main(args) 