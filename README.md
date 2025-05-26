# DeepCAD Implementation

This repository contains an implementation of DeepCAD, a deep learning model for Computer-Aided Design (CAD) operations. The model uses a combination of Graph Neural Networks (GNN) and LSTM to learn and generate CAD operations from input graphs.

## Architecture Overview

The implementation consists of several key components:

1. **Graph Neural Network (GNN)**
   - Uses attention-based message passing
   - 3 GNN layers with ReLU activation
   - Hidden dimension: 256
   - Input dimension: 3 (node features)

2. **LSTM Encoder-Decoder**
   - 2 LSTM layers
   - Hidden dimension: 256
   - Used for operation sequence generation

3. **Prediction Heads**
   - Operation type prediction
   - Parameter prediction
   - Sketch prediction (128-dimensional representation)
   - Node feature prediction

## Hyperparameters

### Model Parameters
- Input dimension: 3
- Hidden dimension: 256
- Latent dimension: 128
- Maximum operations: 20
- Number of operation types: 10
- Number of GCN layers: 3
- Number of LSTM layers: 2

### Training Parameters
- Batch size: 8
- Learning rate: 0.001
- Number of epochs: 100
- Weight decay: 1e-5
- Gradient clipping: 1.0
- Sketch loss weight: 0.1

### Data Parameters
- Maximum nodes per graph: 512
- Number of data loading workers: 4

## Design Choices

1. **Attention Mechanism**
   - Implemented attention in GNN layers for better feature aggregation
   - Uses concatenation of node features for attention computation
   - Applies sigmoid activation for attention weights

2. **Multi-task Learning**
   - Simultaneous prediction of:
     - Operation types
     - Operation parameters
     - Sketch representations
     - Node features
   - Balanced loss weights for different tasks

3. **Graph Processing**
   - Uses adjacency matrix for message passing
   - Implements mask support for variable-sized graphs
   - Three-layer GNN for hierarchical feature extraction

4. **Operation Generation**
   - Greedy decoding for operation sequence generation
   - Special handling for sketch-based operations (extrude, revolve)
   - End-of-sequence detection for variable-length sequences

## Usage

The model can be used for:
1. Learning CAD operations from input graphs
2. Generating CAD programs from new input graphs
3. Predicting operation parameters and sketches

## Dependencies
- PyTorch
- NumPy

## File Structure
- `my_deepcad.py`: Main model implementation
- `config.py`: Configuration and hyperparameters
- `train.py`: Training script
- `data_utils.py`: Data processing utilities
- `generate_sample_data.py`: Data generation utilities 