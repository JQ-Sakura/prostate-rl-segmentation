import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
from rl_module.agents.memory import Memory
import torch.cuda.amp as amp
from contextlib import nullcontext
import gc

class CNNEncoder(nn.Module):
    """Enhanced CNN encoder with attention and residual connections"""
    
    def __init__(self, in_channels: int = 6):
        """
        Args:
            in_channels: Number of input channels (volume + masks)
        """
        super().__init__()
        
        # Define channel attention module
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
        
        # Define spatial attention module
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
        
        # Define residual block
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
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem features
        x = self.stem(x)
        features = []
        for stage in self.encoder_stages:
            x = stage(x)
            features.append(x)
        x = features[-1]
        x = x.view(x.size(0), -1)
        x = self.fusion(x)
        # 不再归一化features，直接返回
        return x

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
        
        # Actor network with more conservative architecture
        self.actor_hidden = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Mean network with tanh activation and scaling
        self.actor_mean = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh()  # 限制输出范围到[-1,1]
        )
        
        # Initialize log_std with very small values and strict limits
        log_std_init = -1.0 * torch.ones(1, action_dim)  # 更小的初始值
        self.actor_log_std = nn.Parameter(log_std_init)
        self.log_std_min = -2  # std≈0.13
        self.log_std_max = 1   # std≈2.7
        
        # Critic network（极简+LayerNorm+收缩）
        self.critic = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Add auxiliary task: predict next state's value
        self.aux_value_pred = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights with smaller values
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 使用更保守的初始化
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, state: Dict) -> Tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
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
            # 只对volume做归一化，mask不归一化
            v = state['volume']
            m = state['masks']
            v = (v - v.mean()) / (v.std() + 1e-6)
            x = torch.cat([v, m], dim=1)
            # Feature extraction
            features = self.encoder(x)
            # 新增：features归一化到每个样本均值0方差1
            features = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + 1e-6)
            print(f'[DEBUG] features mean: {features.mean().item():.4f}, std: {features.std().item():.4f}, min: {features.min().item():.4f}, max: {features.max().item():.4f}')
            # Actor path
            actor_features = self.actor_hidden(features)
            mean = self.actor_mean(actor_features)
            print(f'[DEBUG] mean mean: {mean.mean().item():.4f}, std: {mean.std().item():.4f}')
            # 更严格的log_std限制
            log_std = torch.clamp(self.actor_log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std) + 1e-6
            # 新增：检查mean和std的范围
            if torch.isnan(mean).any() or torch.isnan(std).any():
                print("[ERROR] NaN detected in mean or std")
                mean = torch.zeros_like(mean)
                std = torch.ones_like(std)
            if torch.abs(mean).max() > 1.0:
                print("[WARNING] Mean values out of range, clamping")
                mean = torch.clamp(mean, -1.0, 1.0)
            if torch.abs(std).max() > 1.0:
                print("[WARNING] Std values too large, clamping")
                std = torch.clamp(std, 0.0, 1.0)
            dist = torch.distributions.Normal(mean, std)
            # Critic path
            critic_input = features
            if critic_input.dim() == 1:
                critic_input = critic_input.unsqueeze(0)
            value = self.critic(critic_input)
            value = 2.0 * torch.tanh(value)  # 收缩到[-2,2]
            if torch.isnan(value).any():
                print(f'[ERROR] value nan! features: {features.detach().cpu().numpy()}')
                print(f'[ERROR] value: {value.detach().cpu().numpy()}')
                print(f'[ERROR] critic weights: {[p.detach().cpu().numpy() for p in self.critic.parameters()]}' )
                value = torch.zeros_like(value)
            # aux_value_pred也用features
            aux_value = self.aux_value_pred(features)
            if torch.isnan(mean).any() or torch.isnan(std).any() or torch.isnan(value).any():
                raise ValueError('NaN values detected in network outputs')
            return dist, value, aux_value
        except Exception as e:
            print(f'Forward pass failed: {str(e)}')
            torch.cuda.empty_cache()
            gc.collect()
            raise

class PPOAgent:
    """PPO agent for cancer region detection"""
    
    def __init__(
        self,
        state_dim: Dict,
        action_dim: Any,
        device: str = 'cuda',
        lr: float = 1e-5,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
        memory_batch_size: int = 128,
        gradient_clip: float = 0.1,
        use_amp: bool = True,
        gae_lambda: float = 0.95  # 添加GAE lambda参数
    ):
        """
        Initialize PPO agent
        
        Args:
            state_dim: State space dimensions
            action_dim: Action space dimensions
            device: Device to run on
            lr: Learning rate
            gamma: Discount factor
            epsilon: PPO clipping parameter
            c1: Value loss coefficient
            c2: Entropy coefficient
            memory_batch_size: Batch size for memory
            gradient_clip: Maximum gradient norm
            use_amp: Whether to use automatic mixed precision
            gae_lambda: GAE lambda parameter
        """
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        self.gae_lambda = gae_lambda  # 添加GAE lambda属性
        
        # Add action range limits
        self.action_low = torch.tensor([0, 0, 0]).to(device)
        self.action_high = torch.tensor([127, 127, 127]).to(device)  # Adjust based on voxel size
        
        # Initialize actor-critic network
        self.ac = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(device)
        
        # Use Adam optimizer and add weight decay
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(),
            lr=lr,
            weight_decay=1e-5  # Add L2 regularization
        )
        
        # Initialize memory with larger batch size
        self.memory = Memory(max_batch_size=memory_batch_size)
        
        # Initialize AMP scaler and context
        if use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            self.amp_context = torch.amp.autocast('cuda')
        else:
            self.scaler = None
            self.amp_context = nullcontext()
            
        # Set CUDA performance optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    def select_action(self, state: Any) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Select action and return related information"""
        with torch.no_grad():
            try:
                # Process state input
                if isinstance(state, tuple):
                    state = state[0]
                
                if not isinstance(state, dict):
                    raise TypeError(f"Expected state to be a dict, got {type(state)}")
                
                # Process volume
                volume = state['volume']
                if not isinstance(volume, (np.ndarray, torch.Tensor)):
                    raise TypeError(f"Volume must be ndarray or Tensor, got {type(volume)}")
                
                if isinstance(volume, np.ndarray):
                    volume = torch.FloatTensor(volume)
                
                # Ensure volume is 5D tensor (B, C, H, W, D)
                if volume.dim() == 4:
                    volume = volume.unsqueeze(0)
                elif volume.dim() != 5:
                    raise ValueError(f"Invalid volume dimensions: {volume.shape}")
                
                # Process masks and merge into a tensor
                masks = []
                for mask_name in ['current_mask', 'entropy_map', 'history_mask']:
                    mask = state.get(mask_name, None)
                    if mask is None or not isinstance(mask, (np.ndarray, torch.Tensor)):
                        mask = np.zeros_like(volume[0,0].cpu().numpy())
                    
                    if isinstance(mask, np.ndarray):
                        mask = torch.FloatTensor(mask)
                    
                    # Ensure mask is 4D (B, H, W, D)
                    if mask.dim() == 3:
                        mask = mask.unsqueeze(0)
                    elif mask.dim() == 2:
                        mask = mask.unsqueeze(0).unsqueeze(0)
                    
                    masks.append(mask)
                
                # Stack masks together
                masks_tensor = torch.stack(masks, dim=1)  # (B, 3, H, W, D)
                
                # Create state dictionary
                state_tensors = {
                    'volume': volume.to(self.device),
                    'masks': masks_tensor.to(self.device)
                }
                
                # Forward pass through actor-critic
                dist, value, aux_value = self.ac(state_tensors)
                
                # Calculate entropy to guide exploration
                entropy = dist.entropy().mean()
                self.last_entropy = entropy.item()  # 修复：记录当前熵
                
                # Use entropy and history information to adjust action selection
                history_mask = masks_tensor[:, 2]  # history_mask
                entropy_weight = torch.exp(-3 * history_mask)  # Reduce penalty for history regions
                
                # Sample action
                if np.random.random() < 0.1:
                    action_raw = dist.sample()
                else:
                    action_raw = dist.rsample()
                log_prob = dist.log_prob(action_raw).sum(-1)
                action_scaled = (action_raw + 1) / 2 * (self.action_high - self.action_low) + self.action_low
                action_scaled = torch.clamp(action_scaled, self.action_low, self.action_high)
                if action_scaled.dim() > 1:
                    action_scaled = action_scaled[0]
                if action_raw.dim() > 1:
                    action_raw = action_raw[0]
                return (
                    action_raw.cpu().numpy(),      # 用于存储
                    action_scaled.cpu().numpy(),   # 用于环境
                    log_prob.cpu().numpy()[0],
                    value.cpu().numpy()[0, 0]
                )
            except Exception as e:
                print(f"Warning: Action selection failed with error: {str(e)}")
                random_action = np.random.uniform(
                    low=self.action_low.cpu().numpy(),
                    high=self.action_high.cpu().numpy()
                )
                return random_action, random_action, 0.0, 0.0
        
    def update(
        self,
        batch_size: int = 4,
        epochs: int = 2,
        accumulation_steps: int = 8
    ) -> Dict[str, float]:
        """Update policy with improved stability"""
        try:
            if len(self.memory) == 0:
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0
                }
            
            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Use smaller sub-batches
            sub_batch_size = min(4, len(self.memory) // 4)
            
            # Initialize metrics
            total_policy_loss = 0
            total_value_loss = 0
            total_aux_value_loss = 0
            total_entropy = 0
            num_updates = 0
            
            # Calculate advantages and returns
            with torch.no_grad():
                try:
                    all_states, all_actions, all_rewards, all_values, all_log_probs, all_dones = self.memory.get_batch(self.device)
                    print(f'[DEBUG] reward mean: {all_rewards.mean().item():.4f}, max: {all_rewards.max().item():.4f}')
                    # Calculate GAE
                    advantages = torch.zeros_like(all_rewards)
                    returns = torch.zeros_like(all_rewards)
                    
                    gae = 0
                    for t in reversed(range(len(all_rewards))):
                        if t == len(all_rewards) - 1:
                            next_value = 0
                        else:
                            next_value = all_values[t + 1]
                        
                        delta = all_rewards[t] + self.gamma * next_value * (1 - all_dones[t]) - all_values[t]
                        gae = delta + self.gamma * self.gae_lambda * (1 - all_dones[t]) * gae
                        advantages[t] = gae
                        returns[t] = advantages[t] + all_values[t]
                    
                    # 检查returns/advantages是否有nan
                    if torch.isnan(advantages).any() or torch.isnan(returns).any():
                        print('[ERROR] returns/advantages nan, skip update')
                        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
                    
                    # Normalize advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    print(f'[DEBUG] advantage mean: {advantages.mean().item():.4f}, std: {advantages.std().item():.4f}, max: {advantages.max().item():.4f}')
                    
                    # Move data to CPU
                    advantages = advantages.cpu()
                    returns = returns.cpu()
                    all_log_probs = all_log_probs.cpu()
                    
                    del all_rewards, all_values, all_dones
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        gc.collect()
                        sub_batch_size = max(1, sub_batch_size // 2)
                        print(f"Warning: OOM in advantage computation, reducing sub_batch_size to {sub_batch_size}")
                        return {
                            'policy_loss': 0.0,
                            'value_loss': 0.0,
                            'entropy': 0.0
                        }
                    else:
                        raise e
            
            # Training loop
            for epoch in range(epochs):
                indices = torch.randperm(len(returns))
                
                for start_idx in range(0, len(returns), sub_batch_size):
                    try:
                        # Check memory usage
                        if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.7:
                            torch.cuda.empty_cache()
                            gc.collect()
                        
                        end_idx = min(start_idx + sub_batch_size, len(returns))
                        batch_indices = indices[start_idx:end_idx]
                        
                        # Get batch data
                        batch_data = self.memory.get_batch_indices(batch_indices.numpy(), self.device)
                        batch_states, batch_actions = batch_data[0], batch_data[1]
                        batch_advantages = advantages[batch_indices].to(self.device)
                        batch_returns = returns[batch_indices].to(self.device)
                        batch_old_log_probs = all_log_probs[batch_indices].to(self.device)
                        
                        # Forward pass
                        dist, value, aux_value = self.ac(batch_states)
                        
                        # Critic输出nan检查
                        if torch.isnan(value).any():
                            print('[ERROR] Critic value output nan! Use zeros instead.')
                            value = torch.zeros_like(value)
                        if torch.isnan(aux_value).any():
                            print('[ERROR] Critic aux_value output nan! Use zeros instead.')
                            aux_value = torch.zeros_like(aux_value)
                        
                        # Check NaN
                        if torch.isnan(dist.loc).any() or torch.isnan(dist.scale).any():
                            print("Warning: NaN values detected in distribution parameters")
                            continue
                        
                        entropy = dist.entropy().mean()
                        
                        # Calculate log probabilities and ratio
                        log_probs = dist.log_prob(batch_actions).sum(-1)
                        # 新增：log_prob异常值检查
                        if torch.any(torch.abs(log_probs) > 100):
                            print(f"[ERROR] log_probs too large: min={log_probs.min().item()}, max={log_probs.max().item()}, skip batch")
                            continue
                        if torch.any(torch.abs(batch_old_log_probs) > 100):
                            print(f"[ERROR] batch_old_log_probs too large: min={batch_old_log_probs.min().item()}, max={batch_old_log_probs.max().item()}, skip batch")
                            continue
                        ratio = torch.exp(log_probs - batch_old_log_probs)
                        # 新增：对ratio做clamp，防止极端
                        ratio = torch.clamp(ratio, 1e-4, 1e2)
                        
                        # Calculate losses
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                        
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_1d = value.view(-1)
                        batch_returns_1d = batch_returns.view(-1)
                        aux_value_1d = aux_value.view(-1)
                        value_loss = F.mse_loss(value_1d, batch_returns_1d)
                        aux_value_loss = F.mse_loss(aux_value_1d, batch_returns_1d)
                        # 新增：loss为nan时跳过
                        if torch.isnan(value_loss) or torch.isnan(aux_value_loss):
                            print('[ERROR] value_loss or aux_value_loss nan, skip batch')
                            continue
                        print(f'[DEBUG] policy_loss: {policy_loss.item():.4f}, value_loss: {value_loss.item():.4f}, aux_value_loss: {aux_value_loss.item():.4f}')
                        
                        # Total loss
                        loss = (
                            policy_loss + 
                            self.c1 * value_loss + 
                            0.5 * aux_value_loss -  # Auxiliary task loss weight较小
                            self.c2 * entropy
                        ) / accumulation_steps
                        
                        # policy_loss为nan或极大时跳过
                        if torch.isnan(loss).any() or torch.isnan(policy_loss) or abs(policy_loss.item()) > 1e6:
                            print('Warning: NaN or huge policy_loss detected, skip update')
                            continue
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping and update
                        if (num_updates + 1) % accumulation_steps == 0:
                            if self._check_gradients():  # Add gradient check
                                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.gradient_clip)
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                        
                        # Accumulate metrics
                        total_policy_loss += policy_loss.item() * accumulation_steps
                        total_value_loss += value_loss.item() * accumulation_steps
                        total_aux_value_loss += aux_value_loss.item() * accumulation_steps
                        total_entropy += entropy.item() * accumulation_steps
                        num_updates += 1
                        
                        # Clean up memory
                        del batch_states, batch_actions, batch_advantages, batch_returns
                        del dist, value, aux_value, entropy, log_probs, ratio
                        del policy_loss, value_loss, aux_value_loss, loss
                        torch.cuda.empty_cache()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            gc.collect()
                            sub_batch_size = max(1, sub_batch_size // 2)
                            print(f"Warning: OOM occurred, reducing sub_batch_size to {sub_batch_size}")
                            continue
                        else:
                            raise e
            
            # Clean up memory
            self.memory.clear()
            
            # Calculate average metrics
            num_updates = max(1, num_updates)
            self.last_entropy = total_entropy / num_updates if num_updates > 0 else 0.0
            self.last_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0.0  # 修复：记录policy_loss
            self.last_value_loss = total_value_loss / num_updates if num_updates > 0 else 0.0    # 修复：记录value_loss
            return {
                'policy_loss': total_policy_loss / num_updates,
                'value_loss': total_value_loss / num_updates,
                'aux_value_loss': total_aux_value_loss / num_updates,
                'entropy': total_entropy / num_updates
            }
            
        except Exception as e:
            print(f"Warning: Update failed with error: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0
            }
        
    def _check_gradients(self) -> bool:
        """Check the validity of gradients"""
        for param in self.ac.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print("Warning: NaN gradients detected")
                    return False
                if torch.abs(param.grad).max() > 1000:
                    print("Warning: Gradient explosion detected")
                    return False
        return True
        
    def save(self, path: str):
        """Save model state"""
        state = {
            'actor_critic': self.ac.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)
        
    def load(self, path: str):
        """Load model state"""
        state = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(state['actor_critic'])
        self.optimizer.load_state_dict(state['optimizer'])
        
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict = None):
        """Save a training checkpoint with additional information"""
        state = {
            'epoch': epoch,
            'actor_critic': self.ac.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics or {}
        }
        torch.save(state, path)
        
    def load_checkpoint(self, path: str) -> Dict:
        """Load a training checkpoint and return the saved metrics"""
        state = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(state['actor_critic'])
        self.optimizer.load_state_dict(state['optimizer'])
        return {
            'epoch': state.get('epoch', 0),
            'metrics': state.get('metrics', {})
        } 