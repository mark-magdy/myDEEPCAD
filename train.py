# train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path

from config import Config, get_args
from models.deepcad import DeepCAD
from utils.data_utils import get_dataloader

def masked_mse_loss(pred, target, mask):
    """
    Compute MSE loss with masking
    Args:
        pred: prediction tensor
        target: target tensor
        mask: mask tensor (1 for valid, 0 for padding)
    """
    # Handle different sequence lengths between pred and target
    if pred.size(1) != target.size(1):
        if pred.size(1) > target.size(1):
            # Truncate pred to match target
            pred = pred[:, :target.size(1), :]
        else:
            # Pad target to match pred
            pad_size = pred.size(1) - target.size(1)
            padding = torch.zeros(target.size(0), pad_size, target.size(2), device=target.device)
            target = torch.cat([target, padding], dim=1)
            
            # Also pad the mask
            mask_padding = torch.zeros(mask.size(0), pad_size, device=mask.device)
            mask = torch.cat([mask, mask_padding], dim=1)
    
    # Ensure mask has the right shape
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    
    # Expand mask to match pred shape
    mask = mask.expand_as(pred)
    
    # Compute MSE loss with masking
    loss = ((pred - target) ** 2) * mask
    return loss.sum() / (mask.sum() + 1e-8)

def masked_ce_loss(pred, target, mask):
    """
    Compute cross entropy loss with masking
    Args:
        pred: prediction tensor [B, T, C]
        target: target tensor [B, T]
        mask: mask tensor [B, T]
    """
    # Handle different sequence lengths between pred and target
    if pred.size(1) != target.size(1):
        if pred.size(1) > target.size(1):
            # Truncate pred to match target
            pred = pred[:, :target.size(1), :]
        else:
            # Pad target to match pred
            pad_size = pred.size(1) - target.size(1)
            padding = torch.full((target.size(0), pad_size), -1, dtype=target.dtype, device=target.device)
            target = torch.cat([target, padding], dim=1)
            
            # Also pad the mask
            mask_padding = torch.zeros(mask.size(0), pad_size, device=mask.device)
            mask = torch.cat([mask, mask_padding], dim=1)
    
    # Reshape for cross entropy
    B, T, C = pred.shape
    pred = pred.reshape(-1, C)
    target = target.reshape(-1)
    mask = mask.reshape(-1)
    
    # Compute cross entropy loss
    loss = F.cross_entropy(pred, target.clamp(min=0), reduction='none')
    
    # Apply mask and ignore padded values (-1)
    valid_mask = (target != -1).float() * mask
    loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    return loss

def train_epoch(model, dataloader, optimizer, device, epoch, writer, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_op_type_loss = 0
    total_op_param_loss = 0
    total_sketch_loss = 0
    
    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch['node_features'], batch['mask'], args.teacher_forcing_ratio, batch['node_features'])
        
        # Compute losses
        recon_loss = masked_mse_loss(outputs['output'], batch['node_features'], batch['mask'])
        kl_loss = outputs['kl_div']
        op_type_loss = masked_ce_loss(outputs['op_types'], batch['op_types'], batch['op_mask'])
        op_param_loss = masked_mse_loss(outputs['params'], batch['op_params'], batch['op_mask'])
        sketch_loss = masked_mse_loss(outputs['sketch'], batch['sketch_target'], batch['op_mask'])
        
        # Total loss
        loss = (
            args.recon_weight * recon_loss +
            args.kl_weight * kl_loss +
            args.op_type_weight * op_type_loss +
            args.op_param_weight * op_param_loss +
            args.sketch_weight * sketch_loss
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_op_type_loss += op_type_loss.item()
        total_op_param_loss += op_param_loss.item()
        total_sketch_loss += sketch_loss.item()
        
        # Log metrics
        if (i + 1) % args.log_interval == 0:
            step = epoch * len(dataloader) + i
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/recon_loss', recon_loss.item(), step)
            writer.add_scalar('train/kl_loss', kl_loss.item(), step)
            writer.add_scalar('train/op_type_loss', op_type_loss.item(), step)
            writer.add_scalar('train/op_param_loss', op_param_loss.item(), step)
            writer.add_scalar('train/sketch_loss', sketch_loss.item(), step)
    
    # Compute average losses
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    avg_op_type_loss = total_op_type_loss / len(dataloader)
    avg_op_param_loss = total_op_param_loss / len(dataloader)
    avg_sketch_loss = total_sketch_loss / len(dataloader)
    
    # Log average losses
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    writer.add_scalar('train/epoch_recon_loss', avg_recon_loss, epoch)
    writer.add_scalar('train/epoch_kl_loss', avg_kl_loss, epoch)
    writer.add_scalar('train/epoch_op_type_loss', avg_op_type_loss, epoch)
    writer.add_scalar('train/epoch_op_param_loss', avg_op_param_loss, epoch)
    writer.add_scalar('train/epoch_sketch_loss', avg_sketch_loss, epoch)
    
    return avg_loss

def validate(model, dataloader, device, epoch, writer, args):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_op_type_loss = 0
    total_op_param_loss = 0
    total_sketch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch['node_features'], batch['mask'], 0.0, batch['node_features'])
            
            # Compute losses
            recon_loss = masked_mse_loss(outputs['output'], batch['node_features'], batch['mask'])
            kl_loss = outputs['kl_div']
            op_type_loss = masked_ce_loss(outputs['op_types'], batch['op_types'], batch['op_mask'])
            op_param_loss = masked_mse_loss(outputs['params'], batch['op_params'], batch['op_mask'])
            sketch_loss = masked_mse_loss(outputs['sketch'], batch['sketch_target'], batch['op_mask'])
            
            # Total loss
            loss = (
                args.recon_weight * recon_loss +
                args.kl_weight * kl_loss +
                args.op_type_weight * op_type_loss +
                args.op_param_weight * op_param_loss +
                args.sketch_weight * sketch_loss
            )
            
            # Update metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_op_type_loss += op_type_loss.item()
            total_op_param_loss += op_param_loss.item()
            total_sketch_loss += sketch_loss.item()
    
    # Compute average losses
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    avg_op_type_loss = total_op_type_loss / len(dataloader)
    avg_op_param_loss = total_op_param_loss / len(dataloader)
    avg_sketch_loss = total_sketch_loss / len(dataloader)
    
    # Log average losses
    writer.add_scalar('val/epoch_loss', avg_loss, epoch)
    writer.add_scalar('val/epoch_recon_loss', avg_recon_loss, epoch)
    writer.add_scalar('val/epoch_kl_loss', avg_kl_loss, epoch)
    writer.add_scalar('val/epoch_op_type_loss', avg_op_type_loss, epoch)
    writer.add_scalar('val/epoch_op_param_loss', avg_op_param_loss, epoch)
    writer.add_scalar('val/epoch_sketch_loss', avg_sketch_loss, epoch)
    
    return avg_loss

def train(args):
    """Main training function"""
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir='logs')
    
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
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Create dataloaders
    train_loader = get_dataloader(
        args.data_dir,
        'train',
        batch_size=args.batch_size,
        shuffle=True,
        max_seq_len=args.max_seq_len
    )
    
    val_loader = get_dataloader(
        args.data_dir,
        'val',
        batch_size=args.batch_size,
        shuffle=False,
        max_seq_len=args.max_seq_len
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Early stopping
    patience_counter = 0
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer, args)
        
        # Validate
        val_loss = validate(model, val_loader, device, epoch, writer, args)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Patience: {patience_counter}/{Config.PATIENCE}")
    
    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Train model
    train(args)