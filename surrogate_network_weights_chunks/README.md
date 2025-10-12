# UNet Weights Chunks Manifest

This directory contains the UNet backbone weights split into smaller chunks for easy upload.

## Files:

### Part 1: unet_weights_part_01.pth
- **Size**: 22.49 MB
- **Tensors**: 108
- **Keys**: encoder.stem.0.weight, encoder.stem.0.bias, encoder.stem.1.weight...

### Part 2: unet_weights_part_02.pth
- **Size**: 13.51 MB
- **Tensors**: 7
- **Keys**: encoder.encoder_stages.3.1.weight, encoder.encoder_stages.3.1.bias, encoder.encoder_stages.3.2.weight...

## How to reconstruct:

1. Download all chunk files to the same directory
2. Run the reconstruction script:
   ```bash
   python reconstruct_weights.py
   ```
3. The original weights file will be created in `surrogate_network_weights/`

## Verification:

```python
import torch
weights = torch.load('surrogate_network_weights/unet_backbone_weights.pth')
print(f'Loaded {len(weights)} tensors successfully!')
```
