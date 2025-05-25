# config.py
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="DeepCAD Implementation")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/deepcad', help='Data directory')
    parser.add_argument('--max_nodes', type=int, default=512, help='Maximum number of nodes in a graph')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=3, help='Input dimension for node features')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--max_ops', type=int, default=20, help='Maximum number of operations')
    parser.add_argument('--num_op_types', type=int, default=10, help='Number of operation types')
    parser.add_argument('--num_gcn_layers', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='Number of LSTM layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--sketch_loss_weight', type=float, default=0.1, help='Weight for sketch reconstruction loss')
    
    # Logging and saving
    parser.add_argument('--model_dir', type=str, default='models', help='Model save directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='Log every N batches')
    
    # Hardware
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    return parser.parse_args()

class Config:
    # Model parameters
    NODE_DIM = 3  # Dimension of node features
    HIDDEN_DIM = 256  # Hidden dimension for GNN layers
    NUM_OPS = 5  # Number of operation types
    NUM_PARAMS = 10  # Number of parameters per operation
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Loss weights
    NODE_LOSS_WEIGHT = 1.0
    OP_LOSS_WEIGHT = 1.0
    PARAM_LOSS_WEIGHT = 1.0
    SKETCH_LOSS_WEIGHT = 1.0
    
    # Checkpoint settings
    SAVE_INTERVAL = 10  # Save model every N epochs