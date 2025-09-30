import sys
import os 
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb
import yaml
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from metrics import calculate_metrics
from visualization import plot_3d_comparison, save_all_visualizations
from rl_module.env.cancer_env import CancerEnv
from rl_module.agents.ppo_agent import PPOAgent, CNNEncoder

class ModelEvaluator:
    def __init__(self, model_path, data_dir, output_dir):
        """
        Initialize the evaluator
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.gt_dir = self.data_dir.parent / "gt"  # Real ground truth directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = 200  # Add the maximum number of steps parameter
        
        # Load the configuration file
        config_path = Path("FYP/configs/config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Add the environment configuration
        if 'env' not in self.config:
            self.config['env'] = {
                'similarity_threshold': 0.3,
                'min_region_size': 8,
                'max_region_size': 2000,
                'connectivity': 26,
                'prompt_config': {
                    'distance_threshold': 0.2,
                    'temperature': 0.1,
                    'min_prob': 0.1
                },
                'reward_shaping': {
                    'window_size': 100,
                    'initial_scale': 1.0,
                    'min_scale': 0.1,
                    'max_scale': 2.0,
                    'adaptation_rate': 0.01,
                    'exploration_bonus': 0.01,
                    'entropy_bonus': 0.005,
                    'progress_bonus': 0.02,
                    'disease_bonus': 0.03,
                    'boundary_bonus': 0.01,
                    'use_adaptive_shaping': True
                }
            }
        
        # Initialize the environment
        self.env = CancerEnv(
            data_dir=str(self.data_dir),  # Use the test data directory
            config=self.config,
            model_path=None,  # Evaluation does not require the UNet model
            max_steps=self.max_steps,
            device=str(self.device),
            use_prompt=True
        )
        
        # Initialize the PPO agent
        self.agent = PPOAgent(
            state_dim={'volume': (3, 128, 128, 128), 'masks': (3, 128, 128, 128)},
            action_dim=3,
            device=self.device
        )
        self.agent.ac.encoder = self.create_custom_encoder().to(self.device)
        
        # Load the model
        self.load_model()
        
        # Initialize wandb
        wandb.init(project="cancer-rl-evaluation", name="model_evaluation")
        
    def create_custom_encoder(self):
        # Create a custom CNNEncoder instance that matches the structure of the training code
        class CustomCNNEncoder(CNNEncoder):
            def __init__(self):
                super().__init__(in_channels=6)
                self.in_channels = 6  # 显式设置in_channels
                
                # Define the channel attention module
                class ChannelAttention(nn.Module):
                    def __init__(self, channels: int, reduction: int = 16):
                        super().__init__()
                        self.avg_pool = nn.AdaptiveAvgPool3d(1)
                        self.max_pool = nn.AdaptiveMaxPool3d(1)
                        self.fc = nn.Sequential(
                            nn.Linear(channels, channels // reduction),
                            nn.ReLU(),
                            nn.Linear(channels // reduction, channels)
                        )
                        
                    def forward(self, x):
                        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
                        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
                        out = torch.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1, 1)
                        return x * out
                
                # Define the spatial attention module
                class SpatialAttention(nn.Module):
                    def __init__(self, kernel_size: int = 7):
                        super().__init__()
                        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2)
                        
                    def forward(self, x):
                        avg_out = torch.mean(x, dim=1, keepdim=True)
                        max_out, _ = torch.max(x, dim=1, keepdim=True)
                        x_cat = torch.cat([avg_out, max_out], dim=1)
                        out = torch.sigmoid(self.conv(x_cat))
                        return x * out
                
                # Define the residual block
                class ResBlock(nn.Module):
                    def __init__(self, channels: int):
                        super().__init__()
                        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
                        self.bn1 = nn.BatchNorm3d(channels)
                        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
                        self.bn2 = nn.BatchNorm3d(channels)
                        self.ca = ChannelAttention(channels)
                        self.sa = SpatialAttention()
                        
                    def forward(self, x):
                        residual = x
                        out = F.relu(self.bn1(self.conv1(x)))
                        out = self.bn2(self.conv2(out))
                        out = self.ca(out)
                        out = self.sa(out)
                        out += residual
                        return F.relu(out)
                
                # Stem network
                self.stem = nn.Sequential(
                    nn.Conv3d(self.in_channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm3d(32),
                    nn.ReLU(),
                    nn.MaxPool3d(2)
                )
                
                # Encoder stages
                self.encoder_stages = nn.ModuleList([
                    nn.Sequential(
                        ResBlock(32),
                        nn.Conv3d(32, 64, kernel_size=3, padding=1),
                        nn.BatchNorm3d(64),
                        nn.ReLU(),
                        nn.MaxPool3d(2)
                    ),
                    nn.Sequential(
                        ResBlock(64),
                        nn.Conv3d(64, 128, kernel_size=3, padding=1),
                        nn.BatchNorm3d(128),
                        nn.ReLU(),
                        nn.MaxPool3d(2)
                    ),
                    nn.Sequential(
                        ResBlock(128),
                        nn.Conv3d(128, 256, kernel_size=3, padding=1),
                        nn.BatchNorm3d(256),
                        nn.ReLU(),
                        nn.MaxPool3d(2)
                    ),
                    nn.Sequential(
                        ResBlock(256),
                        nn.Conv3d(256, 512, kernel_size=3, padding=1),
                        nn.BatchNorm3d(512),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool3d((2, 2, 2))
                    )
                ])
                
                # Feature fusion
                self.fusion = nn.Sequential(
                    nn.Linear(512 * 2 * 2 * 2, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(1024, 512),
                    nn.ReLU()
                )
                
            def forward(self, x):
                """Forward pass"""
                print(f"Encoder input shape: {x.shape}")
                
                # Stem features
                x = self.stem(x)
                print(f"After stem shape: {x.shape}")
                
                # Multi-scale feature extraction
                features = []
                for stage in self.encoder_stages:
                    x = stage(x)
                    features.append(x)
                    print(f"After stage shape: {x.shape}")
                
                # Feature fusion
                x = features[-1]
                x = x.view(x.size(0), -1)
                print(f"After flatten shape: {x.shape}")
                
                x = self.fusion(x)
                print(f"Encoder output shape: {x.shape}")
                return x
        
        return CustomCNNEncoder()
        
    def load_model(self):
        """Load the model"""
        try:
            print(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            print("Model loaded successfully")
            
            # Get the state dictionary
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("Get state dictionary from 'model_state_dict' key")
                elif 'actor_critic' in checkpoint:
                    state_dict = checkpoint['actor_critic']
                    print("Get state dictionary from 'actor_critic' key")
                else:
                    state_dict = checkpoint
                    print("Use checkpoint directly as state dictionary")
            else:
                raise ValueError("Invalid checkpoint format")
            
            print("\nKeys in state dictionary:")
            for key in state_dict.keys():
                print(f"- {key}: {state_dict[key].shape if torch.is_tensor(state_dict[key]) else type(state_dict[key])}")
            
            # Load model weights
            missing_keys, unexpected_keys = self.agent.ac.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print("\nMissing keys:")
                for key in missing_keys:
                    print(f"- {key}")
            if unexpected_keys:
                print("\nUnexpected keys:")
                for key in unexpected_keys:
                    print(f"- {key}")
            
            print("\nModel loaded successfully")
            if 'episode' in checkpoint:
                print(f"Model trained {checkpoint['episode']} episodes")
            if 'best_reward' in checkpoint:
                print(f"Best reward: {checkpoint['best_reward']}")
            
            # Set the model to evaluation mode
            self.agent.ac.eval()
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
    def load_modality_data(self, patient_dir):
        """Load the three-modal data for a single patient"""
        modalities = {}
        # Load the data of the three modalities
        modality_files = {
            'adc': 'adc.nii.gz',
            'dwi': 'dwi.nii.gz',
            't2': 't2.nii.gz'
        }
        
        try:
            for modality, filename in modality_files.items():
                file_path = patient_dir / filename
                print(f"Attempting to load {modality} modality data: {file_path}")
                
                if file_path.exists():
                    try:
                        nii_img = nib.load(str(file_path))  # Ensure the path is a string
                        data = nii_img.get_fdata()
                        print(f"{modality} data loaded successfully, shape: {data.shape}")
                        modalities[modality] = data
                    except Exception as e:
                        print(f"Error loading {modality} file: {str(e)}")
                        raise
                else:
                    raise FileNotFoundError(f"Could not find {modality} file: {file_path}")
                    
            # Load the corresponding ground truth
            gt_file = self.gt_dir / f"{patient_dir.name}_l_a1.nii.gz"
            print(f"Attempting to load ground truth: {gt_file}")
            
            if gt_file.exists():
                try:
                    gt_nii = nib.load(str(gt_file))
                    gt_mask = gt_nii.get_fdata()
                    print(f"Ground truth loaded successfully, shape: {gt_mask.shape}")
                except Exception as e:
                    print(f"Error loading ground truth: {str(e)}")
                    raise
            else:
                raise FileNotFoundError(f"Could not find ground truth file: {gt_file}")
                
            return modalities, gt_mask
            
        except Exception as e:
            print(f"Error loading data for {patient_dir}: {str(e)}")
            raise
        
    def evaluate_single_case(self, case_id):
        """Evaluate a single case"""
        try:
            print(f"Starting evaluation for case {case_id}")
            
            # Reset the environment
            print("Resetting environment...")
            state = self.env.reset()[0]
            print(f"Initial state obtained. State keys: {state.keys()}")
            
            # Ensure entropy_map exists
            if 'entropy_map' not in state or state['entropy_map'] is None:
                state['entropy_map'] = np.zeros_like(state['current_mask'])
                self.env.entropy_map = state['entropy_map']
            
            done = False
            truncated = False
            total_reward = 0
            step = 0
            
            while not (done or truncated) and step < self.max_steps:
                # Process the input data
                volume = torch.FloatTensor(state['volume']).unsqueeze(0)
                masks = torch.FloatTensor(np.stack([
                    state['current_mask'],
                    state['entropy_map'],
                    state['history_mask']
                ])).unsqueeze(0)
                
                print("Processed data shapes:")
                print(f"volume shape: {volume.shape}")
                print(f"masks shape: {masks.shape}")
                
                # Move the data to the correct device
                volume = volume.to(self.device)
                masks = masks.to(self.device)
                
                # Model inference
                print("Starting model inference...")
                try:
                    with torch.no_grad():
                        state_dict = {
                            'volume': volume,
                            'masks': masks
                        }
                        # Ignore the third return value (None)
                        action_dist, value = self.agent.ac(state_dict)[:2]
                        action = action_dist.mean
                        print(f"Action value: {action.cpu().numpy()[0]}")
                        
                    # Execute the action
                    next_state, reward, done, truncated, info = self.env.step(action.cpu().numpy()[0])
                    total_reward += reward
                    state = next_state
                    step += 1
                    
                except Exception as e:
                    print(f"Model inference failed: {str(e)}")
                    raise
                
            return state['current_mask'], info, reward, state
            
        except Exception as e:
            print(f"Error processing data or model inference: {str(e)}")
            raise
        
    def evaluate_all(self):
        """Evaluate all test data"""
        test_cases = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if not test_cases:
            print(f"\nWarning: No test data found in {self.data_dir}")
            print("Please ensure the test data directory contains the following structure:")
            print("evaluation/data/test/")
            print("  ├── patient1/")
            print("  │   ├── adc.nii.gz")
            print("  │   ├── dwi.nii.gz")
            print("  │   └── t2.nii.gz")
            print("  └── patient2/")
            print("      ├── adc.nii.gz")
            print("      ├── dwi.nii.gz")
            print("      └── t2.nii.gz")
            print("\nevaluation/data/gt/")
            print("  ├── patient1_l_a1.nii.gz")
            print("  └── patient2_l_a1.nii.gz")
            return {}
            
        all_metrics = []
        
        for patient_dir in tqdm(test_cases, desc="Evaluating cases"):
            try:
                # Evaluate a single case
                pred_mask, metrics, reward, modalities = self.evaluate_single_case(patient_dir)
                
                # Save the results
                case_name = patient_dir.name
                metrics["case_name"] = case_name
                metrics["total_reward"] = reward
                all_metrics.append(metrics)
                
                # Record to wandb
                wandb.log({
                    "case": case_name,
                    **metrics
                })
                
                # Save all visualization results
                save_all_visualizations(
                    modalities['t2'],  # Use T2 as reference for display
                    pred_mask,
                    self.env.gt_mask,
                    case_name,
                    self.output_dir / case_name
                )
                
                # Save the predicted mask
                pred_nii = nib.Nifti1Image(pred_mask, np.eye(4))
                nib.save(pred_nii, self.output_dir / f"{case_name}_pred.nii.gz")
                
            except Exception as e:
                print(f"Error processing case {patient_dir}: {str(e)}")
                continue
            
        if not all_metrics:
            print("No cases were successfully evaluated!")
            return {}
            
        # Calculate and save overall metrics
        mean_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
            if key not in ["case_name"]
        }
        
        # Save the evaluation report
        self.save_evaluation_report(all_metrics, mean_metrics)
        
        return mean_metrics
        
    def save_evaluation_report(self, all_metrics, mean_metrics):
        """Save the evaluation report"""
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, "w") as f:
            f.write("=== Cancer Region Detection Model Evaluation Report ===\n\n")
            
            # Write overall metrics
            f.write("Overall Metrics:\n")
            for metric, value in mean_metrics.items():
                if isinstance(value, (list, tuple)):
                    # If it's a list type, calculate the average
                    avg_value = sum(value) / len(value) if value else 0.0
                    f.write(f"{metric}: {avg_value:.4f} (averaged)\n")
                elif isinstance(value, (int, float)):
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: {str(value)}\n")
            f.write("\n")
            
            # Write individual case metrics
            f.write("Individual Case Metrics:\n")
            for metrics in all_metrics:
                f.write(f"\nCase: {metrics['case_name']}\n")
                for metric, value in metrics.items():
                    if metric != "case_name":
                        if isinstance(value, (list, tuple)):
                            # If it's a list type, calculate the average
                            avg_value = sum(value) / len(value) if value else 0.0
                            f.write(f"{metric}: {avg_value:.4f} (averaged)\n")
                        elif isinstance(value, (int, float)):
                            f.write(f"{metric}: {value:.4f}\n")
                        else:
                            f.write(f"{metric}: {str(value)}\n")
                        
        print(f"Evaluation report saved to {report_path}")

class ActorCritic(nn.Module):
    """Combined actor-critic network"""
    
    def __init__(self, state_dim: Dict, action_dim: Any = 3):
        super().__init__()
        
        # Get action dimension
        if isinstance(action_dim, (tuple, list, np.ndarray)):
            action_dim = 3  # Default to 3 for (x, y, z) coordinates
        elif hasattr(action_dim, 'shape'):
            action_dim = 3  # Default to 3 for Box space
        
        # Encoder for processing state
        self.encoder = CNNEncoder(in_channels=6)  # 3 modalities + 3 masks
        
        # Actor hidden layers
        self.actor_hidden = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor mean layer
        self.actor_mean = nn.Linear(256, action_dim)
        
        # Use learnable log_std
        log_std_init = -1.0 * torch.ones(1, action_dim)
        self.actor_log_std = nn.Parameter(log_std_init)
        
        # Critic layer
        self.critic = nn.Linear(256, 1)
        
    def forward(self, state: Dict) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        """Forward pass"""
        try:
            # Input check and preprocessing
            if not isinstance(state, dict):
                raise ValueError(f"Expected state to be a dict, got {type(state)}")
            
            if 'volume' not in state:
                raise ValueError("State must contain 'volume' key")
                
            if 'masks' not in state:
                if isinstance(state['volume'], torch.Tensor):
                    state['masks'] = torch.zeros_like(state['volume'])
                else:
                    raise ValueError("Volume must be a torch.Tensor")
            
            # Dimension check
            if state['volume'].dim() != 5:
                raise ValueError(f"Volume must be 5D (B,C,H,W,D), got shape {state['volume'].shape}")
            if state['masks'].dim() != 5:
                raise ValueError(f"Masks must be 5D (B,C,H,W,D), got shape {state['masks'].shape}")
            
            # Concatenate volume and masks
            x = torch.cat([state['volume'], state['masks']], dim=1)
            print(f"Input concatenated shape: {x.shape}")
            
            # Feature extraction
            features = self.encoder(x)
            print(f"Features shape: {features.shape}")
            
            # Actor path
            actor_features = self.actor_hidden(features)
            print(f"Actor hidden features shape: {actor_features.shape}")
            
            mean = self.actor_mean(actor_features)
            print(f"Actor mean shape: {mean.shape}")
            
            # Use learnable log_std
            std = torch.exp(self.actor_log_std) + 1e-6
            print(f"Actor std shape: {std.shape}")
            
            # Use normal distribution
            dist = torch.distributions.Normal(mean, std)
            
            # Critic path
            value = self.critic(actor_features)
            print(f"Critic value shape: {value.shape}")
            
            # Check the validity of the output
            if torch.isnan(mean).any() or torch.isnan(std).any() or torch.isnan(value).any():
                raise ValueError("NaN values detected in network outputs")
            
            return dist, value, None
            
        except Exception as e:
            print(f"Forward pass failed: {str(e)}")
            raise

def main():
    # Use the final model in the logs_bk directory
    model_path = "FYP/logs_bk/run_20250211_003345/final_model.pth"
    data_dir = "FYP/evaluation/data/test"
    output_dir = "FYP/evaluation/results"
    
    # Check if the directories exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Test data directory not found: {data_dir}")
    if not os.path.exists(os.path.dirname(data_dir) + "/gt"):
        raise FileNotFoundError(f"Ground truth directory not found: {os.path.dirname(data_dir)}/gt")
        
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the evaluator and run the evaluation
    evaluator = ModelEvaluator(model_path, data_dir, output_dir)
    mean_metrics = evaluator.evaluate_all()
    
    # Print overall results
    print("\nEvaluation Results:")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main() 