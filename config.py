# config.py
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description="DeepCAD Implementation")
    
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
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='Teacher forcing ratio')
    
    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--kl_weight', type=float, default=0.1, help='KL divergence weight')
    parser.add_argument('--op_type_weight', type=float, default=1.0, help='Operation type loss weight')
    parser.add_argument('--op_param_weight', type=float, default=1.0, help='Operation parameter loss weight')
    parser.add_argument('--sketch_weight', type=float, default=0.5, help='Sketch loss weight')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    
    return parser.parse_args()

class Config:
    # Data parameters
    DATA_DIR = 'data/deepcad'
    MAX_SEQ_LEN = 100
    
    # Model parameters
    INPUT_DIM = 3
    HIDDEN_DIM = 256
    LATENT_DIM = 128
    NUM_LAYERS = 3
    NUM_HEADS = 8
    DROPOUT = 0.1
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 100
    GRAD_CLIP = 1.0
    TEACHER_FORCING_RATIO = 0.5
    
    # Loss weights
    RECON_WEIGHT = 1.0
    KL_WEIGHT = 0.1
    OP_TYPE_WEIGHT = 1.0
    OP_PARAM_WEIGHT = 1.0
    SKETCH_WEIGHT = 0.5
    
    # Logging and saving
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 10
    CHECKPOINT_DIR = 'checkpoints'
    
    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0
    
    # Model architecture
    NUM_OP_TYPES = 10
    NUM_OP_PARAMS = 10
    SKETCH_DIM = 128
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 1e-4