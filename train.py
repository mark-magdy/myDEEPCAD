# train.py
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from config import Config
from data_utils import get_dataloader
from my_deepcad import DeepCAD

def train(args):
    # Create log directory
    os.makedirs('logs', exist_ok=True)
    writer = SummaryWriter('logs')
    
    # Initialize model
    model = DeepCAD(
        node_dim=Config.NODE_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_ops=Config.NUM_OPS,
        num_params=Config.NUM_PARAMS
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Initialize dataloaders
    train_loader = get_dataloader(args.data_dir, 'train', batch_size=Config.BATCH_SIZE)
    val_loader = get_dataloader(args.data_dir, 'val', batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        # Training
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS}'):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate losses
            node_loss = nn.MSELoss()(outputs['node_pred'], batch['node_features'])
            op_loss = nn.CrossEntropyLoss()(outputs['op_pred'], batch['op_types'])
            param_loss = nn.MSELoss()(outputs['param_pred'], batch['op_params'])
            sketch_loss = nn.MSELoss()(outputs['sketch_pred'], batch['sketch_target'])
            
            # Total loss
            loss = (Config.NODE_LOSS_WEIGHT * node_loss +
                   Config.OP_LOSS_WEIGHT * op_loss +
                   Config.PARAM_LOSS_WEIGHT * param_loss +
                   Config.SKETCH_LOSS_WEIGHT * sketch_loss)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        
        # Log training metrics
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/node', node_loss.item(), epoch)
        writer.add_scalar('Loss/op', op_loss.item(), epoch)
        writer.add_scalar('Loss/param', param_loss.item(), epoch)
        writer.add_scalar('Loss/sketch', sketch_loss.item(), epoch)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch)
                
                # Calculate validation losses
                node_loss = nn.MSELoss()(outputs['node_pred'], batch['node_features'])
                op_loss = nn.CrossEntropyLoss()(outputs['op_pred'], batch['op_types'])
                param_loss = nn.MSELoss()(outputs['param_pred'], batch['op_params'])
                sketch_loss = nn.MSELoss()(outputs['sketch_pred'], batch['sketch_target'])
                
                loss = (Config.NODE_LOSS_WEIGHT * node_loss +
                       Config.OP_LOSS_WEIGHT * op_loss +
                       Config.PARAM_LOSS_WEIGHT * param_loss +
                       Config.SKETCH_LOSS_WEIGHT * sketch_loss)
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        # Save checkpoint
        if (epoch + 1) % Config.SAVE_INTERVAL == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pt')
        
        print(f'Epoch {epoch+1}/{Config.NUM_EPOCHS}:')
        print(f'Training Loss: {avg_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/deepcad',
                      help='Directory containing the dataset')
    args = parser.parse_args()
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    train(args)