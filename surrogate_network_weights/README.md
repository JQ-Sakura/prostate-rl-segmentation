# Surrogate Network Weights

This directory contains the extracted UNet backbone weights from the trained RL-PromptSeg model.

## Contents

- `unet_backbone_weights.pth`: Extracted UNet backbone weights (115 weight tensors)
- `weights_summary.txt`: Detailed summary of all weight tensors and their shapes

## Model Architecture

The extracted weights correspond to the UNet backbone components:

- **Encoder Stem**: Initial 3D convolutional layers with batch normalization
- **Encoder Stages**: 4-stage encoder with attention mechanisms (Channel Attention and Spatial Attention)
- **Bottleneck**: Deepest feature extraction layer
- **Decoder Stages**: 4-stage decoder with skip connections
- **Final Layer**: Output segmentation layer

## Key Features

- **Input Channels**: 6 (ADC, DWI, T2, gland, and 2 additional channels)
- **Base Filters**: 32
- **Attention Mechanisms**: 
  - Channel Attention (CA) blocks
  - Spatial Attention (SA) blocks
- **Normalization**: Batch normalization throughout the network

## Usage

To load the UNet backbone weights:

```python
import torch
from rl_module.models.unet3d import UNet3D

# Load the extracted weights
weights = torch.load('unet_backbone_weights.pth', map_location='cpu')

# Create UNet model
model = UNet3D(
    in_channels=6,
    base_filters=32,
    num_levels=4,
    use_se=True,
    use_deep_supervision=True,
    dropout=0.1
)

# Load the weights
model.load_state_dict(weights, strict=False)
```

## Weight Statistics

- **Total Weight Tensors**: 115
- **Source Model**: `best_model_episode_0.pth`
- **Model Size**: ~50MB (estimated)
- **Architecture**: 3D UNet with attention mechanisms

## Components Included

### Encoder Components
- Stem layers (initial feature extraction)
- 4 encoder stages with increasing channel dimensions (32→64→128→256→512)
- Channel and spatial attention blocks
- Batch normalization layers

### Decoder Components  
- Skip connections from encoder stages
- Upsampling layers
- Feature fusion layers

### Attention Mechanisms
- Channel Attention (CA): Focuses on important feature channels
- Spatial Attention (SA): Focuses on important spatial locations
- Both attention mechanisms are included in the extracted weights

## Notes

- These weights were extracted from a fully trained RL-PromptSeg model
- The weights include all necessary components for 3D prostate cancer segmentation
- Batch normalization running statistics are preserved for inference
- The model is optimized for multi-parametric MRI data (ADC, DWI, T2, gland)
