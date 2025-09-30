# Prostate Cancer Segmentation with Reinforcement Learning and 3D U-Net

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
# from repo root
pip install -r requirements.txt
```

If `requirements.txt` is not yet present, install common dependencies manually:

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

## Saving and Viewing Results

- Predictions, ground truths, and metadata are stored under `training_results/`.
- Use `view_saved_results.py` to inspect saved runs.
- `README_result_saver.md` describes how results are recorded.

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

## GitHub: How to Push This Repository

If you’re using GitHub CLI (recommended):

```bash
# initialize (if not already)
git init
git add .
git commit -m "Initial commit"

# create and push repo (public)
gh repo create --public --source . --remote origin --push
```

Without GitHub CLI, create a new repo on GitHub via the web UI, then:

```bash
git remote add origin https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
git branch -M main
git push -u origin main
```

## Collaborators

Add collaborator `s-sd` with write access using GitHub CLI:

```bash
gh repo add-collaborator s-sd --permission push
```

Alternatively, use the GitHub web UI: Settings → Collaborators → Add people → `s-sd` or invite by email `shaheersd@gmail.com`.

## License

Provide your chosen license here (e.g., MIT). If unsure, consider MIT for permissive use.
