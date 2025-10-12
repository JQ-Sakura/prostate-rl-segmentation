# Promptable segmentation enables minimal-effort expert-level prostate cancer delineation

## Authors

Junqing Yang¹, Natasha Thorley¹,², Ahmed Nadeem Abbasi³, Shonit Punwani¹,², Zion Tse⁴, Yipeng Hu¹, Shaheer U. Saeed¹,⁴*

### Affiliations

¹ UCL Hawkes Institute; Department of Medical Physics and Biomedical Engineering, University College London, UK  
² Centre of Medical Imaging, University College London, UK  
³ Department of Oncology, Aga Khan University Hospital, Pakistan  
⁴ Centre for Bioengineering; School of Engineering and Materials Science, Queen Mary University of London, UK  

*Corresponding author: shaheer.saeed@qmul.ac.uk

## About

This repository contains code for training and evaluating a reinforcement-learning-assisted 3D medical image segmentation model. It includes:

- RL training loop and agents (PPO)
- 3D U-Net backbone
- Custom environment and reward shaping
- Evaluation utilities, metrics, and visualization
- Result saving and leaderboard of top runs

## Environment and Requirements

- OS: Windows 10/11, Linux, or macOS
- Python: 3.9+ recommended
- GPU: CUDA-capable GPU recommended for training

Install Python dependencies (create a virtual environment if desired):

```bash
pip install numpy scipy torch torchvision torchaudio nibabel scikit-image scikit-learn tqdm pyyaml matplotlib SimpleITK
```

## Repository Structure

- `run.py`: Main entrypoint to run training/evaluation pipelines depending on config flags.
- `rl_module/`: RL components
  - `rl_module/rl_train.py`: RL training loop
  - `rl_module/agents/`: PPO agent and memory
  - `rl_module/env/`: Environment and region growing logic
  - `rl_module/models/`: 3D U-Net
  - `rl_module/utils/`: losses, EMA, post-processing, prompt segmentation, rewards, evaluation, result saver
  - `rl_module/configs/train_config.yaml`: Training configuration
- `evaluation/`: Evaluation scripts, metrics, visualization, and example data
  - `evaluation/evaluate.py`: Batch evaluation
  - `evaluation/metrics.py`: Metrics (e.g., Dice)
  - `evaluation/visualization.py`: Rendering utilities
  - `evaluation/data/`: Example test and GT data structure
- `training_results/`: Saved predictions, GT, metadata, and volumes
- `surrogate_network_weights/`: Extracted UNet backbone weights for standalone usage
  - `unet_backbone_weights.pth`: Pre-trained UNet weights (115 tensors, 37MB)
  - `weights_summary.txt`: Detailed weight tensor summary
  - `README.md`: Usage instructions and architecture details
- `surrogate_network_weights_chunks/`: Split weights for easy upload (each <25MB)
  - `unet_weights_part_01.pth`: First part (22.5MB, 108 tensors)
  - `unet_weights_part_02.pth`: Second part (13.5MB, 7 tensors)
  - `reconstruct_weights.py`: Script to rebuild original weights
  - `README.md`: Chunk reconstruction instructions
- `configs/`: High-level configs

## Quick Start

1) Prepare data following the structure in `evaluation/data/`.
2) Adjust configuration files in `configs/` and `rl_module/configs/train_config.yaml` as needed.
3) Run training/evaluation:

```bash
# Example: run main pipeline
python run.py

# Example: RL training directly
python rl_module/rl_train.py --config rl_module/configs/train_config.yaml

# Example: Evaluate saved predictions
python evaluation/evaluate.py --results_dir training_results --gt_dir evaluation/data/gt
```

Common optional flags can be viewed with `-h` on each script.

## Git LFS Setup

This repository uses Git LFS (Large File Storage) to manage large model files:

- **Pre-trained weights**: `surrogate_network_weights/unet_backbone_weights.pth` (37MB)
- **Medical images**: `.nii.gz` files in evaluation data

### First-time setup:
```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone the repository
git clone https://github.com/yourusername/prostate-rl-segmentation.git
cd prostate-rl-segmentation

# Pull large files
git lfs pull
```

### For existing repositories:
```bash
# Pull latest LFS files
git lfs pull

# Check LFS file status
git lfs ls-files
```

## Saving and Viewing Results

- Predictions, ground truths, and metadata are stored under `training_results/`.
- Use `view_saved_results.py` to inspect saved runs.
- `README_result_saver.md` describes how results are recorded.

## Using Pre-trained Weights

The repository includes pre-trained UNet backbone weights in two formats:

### Split Weights
- **Location**: `surrogate_network_weights_chunks/` (each part <25MB)
- **Reconstruction**:
  ```bash
  python surrogate_network_weights_chunks/reconstruct_weights.py
  ```
- **Benefits**: Easy upload to GitHub, no size restrictions

### Load Weights in Code
```python
import torch
from rl_module.models.unet3d import UNet3D

# Load weights
weights = torch.load('surrogate_network_weights/unet_backbone_weights.pth', map_location='cpu')

# Create model
model = UNet3D(in_channels=6, base_filters=32, num_levels=4, use_se=True)
model.load_state_dict(weights, strict=False)
print('UNet model loaded successfully!')
```

## Configuration

- Global configs: `configs/config.yaml`
- RL training configs: `rl_module/configs/train_config.yaml`

Tune batch sizes, learning rates, reward shaping, and UNet depth in these files.

## Reproducing Evaluation

The `evaluation/` folder includes scripts for computing metrics and generating reports. Example command:

```bash
python evaluation/evaluate.py \
  --pred_dir training_results/predictions \
  --gt_dir evaluation/data/gt \
  --report_file evaluation/results/evaluation_report.txt
```

## Results

### Performance Comparison

| Model | PROMIS | PICAI |
|-------|--------|-------|
| SAM | 0.236 ± 0.107 | 0.294 ± 0.132 |
| MedSAM | 0.267 ± 0.138 | 0.342 ± 0.142 |
| Combiner | 0.330 ± 0.180 | 0.469 ± 0.156 |
| T2-predictor | 0.339 ± 0.192 | 0.394 ± 0.141 |
| UniverSeg | 0.307 ± 0.216 | 0.351 ± 0.154 |
| UNet | 0.327 ± 0.198 | 0.426 ± 0.153 |
| nnUNet | 0.414 ± 0.201 | 0.461 ± 0.137 |
| Swin-UNeTr | 0.427 ± 0.185 | 0.477 ± 0.133 |
| Human | 0.531 ± 0.941 | - |
| **RL-PromptSeg** | **0.526 ± 0.112** | **0.566 ± 0.139** |

**RL-PromptSeg** significantly outperforms the previous state-of-the-art fully-automated method Swin-UNeTr by 9.9% and 8.9% percentage points on PROMIS and PICAI datasets respectively, while showing comparable performance to human observers.

### Qualitative Results

![Samples from PROMIS segmented using our RL-PromptSeg approach](Samples%20from%20PROMIS%20segmented%20using%20our%20RL-PromptSeg%20approach.png)

Visual comparison showing RL-PromptSeg predictions alongside ground truth annotations on PROMIS dataset samples. The model successfully identifies lesion locations and general extent, achieving a 10X reduction in annotation time (131 seconds vs 1093 seconds per case) while maintaining expert-level accuracy.
