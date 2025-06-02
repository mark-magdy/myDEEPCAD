# DeepCAD: Deep Generative Network for Computer-Aided Design Models

This repository contains an implementation of DeepCAD, a deep generative network for Computer-Aided Design (CAD) models. The implementation is based on the paper ["DeepCAD: A Deep Generative Network for Computer-Aided Design Models"](https://www.cs.columbia.edu/cg/deepcad/) by Rundi Wu, Chang Xiao, and Changxi Zheng (ICCV 2021).

## Overview

DeepCAD is a deep learning model that can:
1. Encode CAD models into a compact latent representation
2. Decode latent vectors back into CAD models
3. Generate new CAD models by sampling from the latent space

Our implementation uses a transformer-based architecture for both the encoder and decoder, allowing the model to effectively handle variable-length sequences of CAD operations.

## Hyperparameters

Key hyperparameters:
- Input dimension: 3 (x, y, z coordinates)
- Hidden dimension: 256
- Latent dimension: 128
- Number of transformer layers: 3
- Number of attention heads: 8
- Dropout rate: 0.1
- Learning rate: 0.0001
- Batch size: 16
- Weight decay: 1e-5

## Project Structure

```
.
├── models/
│   └── deepcad.py         # DeepCAD model implementation
├── utils/
│   └── data_utils.py      # Data utilities
├── data/
│   └── deepcad/           # Dataset directory
│       ├── train/         # Training data
│       ├── val/           # Validation data
│       └── test/          # Test data
├── checkpoints/           # Model checkpoints
├── config.py              # Configuration
├── generate_sample_data.py # Script to generate sample data
├── train.py               # Training script
├── test.py                # Testing script
├── evaluate.py            # Evaluation script
└── README.md              # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/deepcad.git
cd deepcad
```

2. Install dependencies:
```bash
pip install torch numpy matplotlib tqdm tensorboard
```

## Data Generation

To generate sample data for training and testing:

```bash
python generate_sample_data.py
```

This will create sample CAD data in the `data/deepcad` directory.

## Training

To train the model:

```bash
python train.py --data_dir data/deepcad
```

Additional training options:
```
--batch_size: Batch size (default: 32)
--lr: Learning rate (default: 0.0001)
--num_epochs: Number of epochs (default: 100)
--hidden_dim: Hidden dimension (default: 256)
--latent_dim: Latent dimension (default: 128)
--resume: Resume training from checkpoint
```

## Testing

To test the model:

```bash
python test.py --mode generate --checkpoint checkpoints/best_model.pt --num_samples 10
```

Available test modes:
```
encode: Encode the dataset to latent space
reconstruct: Reconstruct the dataset
decode: Decode latent vectors to CAD models
generate: Generate new CAD models
```

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

Additional evaluation options:
```
--generate_samples: Generate new samples
--num_samples: Number of samples to generate (default: 10)
--output_dir: Output directory for evaluation results (default: evaluation_results)
```

## Model Architecture

The DeepCAD model consists of:

1. **Encoder**: Transformer-based encoder that processes CAD models and produces a latent vector representation.
2. **Decoder**: Transformer-based decoder that generates CAD operations from a latent vector.

The model is trained as a variational autoencoder (VAE), with additional losses for operation type prediction, parameter regression, and sketch feature generation.

## Implementation Status

The implementation is fully functional and includes:
- Transformer-based encoder-decoder architecture
- Proper handling of variable-length sequences with padding and masking
- Multi-task learning for operation type prediction, parameter regression, and sketch generation
- Training, testing, and evaluation scripts
- Sample data generation

## Comparing with Original DeepCAD

This implementation differs from the original DeepCAD paper in several key aspects:

1. **Architecture**:
   - This implementation: Uses transformer-based architecture for both encoder and decoder
   - Original: Uses graph neural networks (GNNs) for encoding and LSTMs for decoding

2. **Model Structure**:
   - This version: Implements a variational autoencoder with transformer blocks and self-attention
   - Original: Uses a hierarchical encoder-decoder with graph message passing and RNN-based sequence generation

3. **Data Representation**:
   - This implementation: Focuses on node features, adjacency matrices, and operation sequences
   - Original: Uses more specialized CAD-specific representations with detailed sketch processing

4. **Training Approach**:
   - This version: Uses multi-task learning with combined losses for reconstruction, operation prediction, etc.
   - Original: Implements more specialized training pipeline with separate modules for sketch and operation handling

5. **Generation Process**:
   - This implementation: Generates directly from latent space
   - Original: Uses more structured generation with specific constraints for CAD validity

## Comparing Model Outputs

To compare outputs between this implementation and the original DeepCAD:

1. Generate samples with our implementation:
```bash
python test.py --mode generate --checkpoint checkpoints/best_model.pt --num_samples 10 --output_dir comparison/ours
```

2. Download original DeepCAD samples (if available):
```bash
# This would require access to original DeepCAD samples
# You can download them from the official repository or request from the authors
```

3. Visualize and compare results:
```bash
python compare_results.py --our_dir comparison/ours --original_dir comparison/original
```

Note: For a proper comparison, you would need to:
1. Obtain samples from the original DeepCAD implementation
2. Convert both outputs to a common format (e.g., .step or .obj files)
3. Use metrics like Chamfer distance or IoU for quantitative comparison
4. Perform user studies for qualitative evaluation 