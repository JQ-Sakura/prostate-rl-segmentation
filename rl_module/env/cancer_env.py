import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, Any
from gymnasium import spaces
from scipy.ndimage import zoom
from scipy.ndimage import label, generate_binary_structure, binary_dilation
from rl_module.data.dataset import MRIDataset
from rl_module.env.region_growing import RegionGrower
from rl_module.models.unet3d import UNet3D
from rl_module.utils.prompt_segmentation import PromptSegmentation
from rl_module.utils.post_processing import PostProcessor
from rl_module.utils.adaptive_reward import AdaptiveRewardShaper
import torch
import random
from collections import deque
import signal

STANDARD_SIZE = (128, 128, 128)  # Standardized image size

class CancerEnv(gym.Env):
    """
    Gymnasium environment for cancer region detection
    
    State space:
        - 3D MRI data (3 modalities)
        - Current segmentation mask
        - Entropy map
        - History mask (previously selected regions)
        
    Action space:
        - Seed point selection (z, y, x)
    """
    
    def __init__(
        self,
        data_dir: str,
        config: Dict,
        model_path: Optional[str] = None,
        max_steps: int = 5,
        device: str = 'cuda',
        use_prompt: bool = True,
        dataset_type: str = 'all'
    ):
        """
        Initialize environment
        
        Args:
            data_dir: Directory containing MRI data
            config: Configuration dictionary
            model_path: Path to pretrained UNet model
            max_steps: Maximum steps per episode
            device: Device to run model on
            use_prompt: Whether to use prompt-based segmentation
            dataset_type: Dataset type ('train', 'valid', 'test', or 'all')
        """
        super().__init__()
        
        # Initialize basic attributes
        self.data_dir = data_dir
        self.config = config
        self.device = device
        self.max_steps = max_steps
        self.use_prompt = use_prompt
        self.dataset_type = dataset_type
        
        # Initialize metrics related attributes
        self.current_iou = 0.0
        self.current_dice = 0.0
        self.best_iou = 0.0
        self.best_dice = 0.0
        self.episode_iou_history = []
        self.episode_dice_history = []
        self.dice_history = []  # Initialize dice history record list
        
        # Performance metrics
        self.raw_metrics = {'iou': 0.0, 'dice': 0.0}
        self.post_metrics = {'iou': 0.0, 'dice': 0.0}
        self.boundary_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Initialize other components
        # 判断是否为PICAI新数据集
        data_cfg = config.get('data', {})
        if all([
            'picai_folds' in data_cfg,
            'mask_dir' in data_cfg,
            'keep_cases_path' in data_cfg
        ]):
            self.dataset = MRIDataset(
                data_dirs=data_cfg['picai_folds'],
                mask_dir=data_cfg['mask_dir'],
                keep_cases_path=data_cfg['keep_cases_path'],
                dataset_type=dataset_type,
                split_ratio=data_cfg.get('split_ratio', {'train': 0.7, 'valid': 0.15, 'test': 0.15}),
                seed=config.get('training', {}).get('seed', 42)
            )
        else:
            self.dataset = MRIDataset(
                data_dir=data_dir, 
                dataset_type=dataset_type, 
                split_ratio=config.get('data', {}).get('split_ratio', {'train': 0.7, 'valid': 0.15, 'test': 0.15}),
                seed=config.get('training', {}).get('seed', 42)
            )
        self.region_grower = RegionGrower(
            similarity_threshold=config['env']['similarity_threshold'],
            min_size=config['env']['min_region_size'],
            max_size=config['env']['max_region_size'],
            connectivity=config['env']['connectivity']
        )
        
        # Initialize post-processor
        self.post_processor = PostProcessor(
            min_size=config['env'].get('min_region_size', 64),
            apply_crf=True,
            crf_iterations=5,
            crf_config={
                'bilateral_sxy': (5, 5),
                'bilateral_srgb': (13, 13, 13),
                'bilateral_compat': 10,
                'gaussian_sxy': (3, 3),
                'gaussian_compat': 3
            }
        )
        
        # Initialize prompt segmentation
        if use_prompt:
            self.prompt_segmentation = PromptSegmentation(
                distance_threshold=config['env']['prompt_config']['distance_threshold'],
                temperature=config['env']['prompt_config']['temperature'],
                min_prob=config['env']['prompt_config']['min_prob'],
                device=device
            )
            
        # Initialize reward shaper
        self.reward_shaper = AdaptiveRewardShaper(
            window_size=config['env']['reward_shaping'].get('window_size', 100),
            initial_scale=config['env']['reward_shaping'].get('initial_scale', 1.0),
            min_scale=config['env']['reward_shaping'].get('min_scale', 0.1),
            max_scale=config['env']['reward_shaping'].get('max_scale', 2.0),
            adaptation_rate=config['env']['reward_shaping'].get('adaptation_rate', 0.01)
        )
        
        # Initialize action space and observation space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            'volume': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,) + STANDARD_SIZE,
                dtype=np.float32
            ),
            'masks': spaces.Box(
                low=0,
                high=1,
                shape=(3,) + STANDARD_SIZE,
                dtype=np.float32
            )
        })
        
        # Initialize state variables
        self.current_volume = None
        self.current_mask = None
        self.gt_mask = None
        self.steps_taken = 0
        self.volume_shape = None
        self.history_mask = None
        self.entropy_map = None
        self.model = None  # 提前初始化，防止后续引用报错
        self.current_idx = 0  # 添加当前数据索引跟踪
        
        # Initialize volume shape
        self.volume_shape = STANDARD_SIZE
        
        # Initialize masks with same shape as volume
        self.current_mask = np.zeros(self.volume_shape, dtype=np.float32)
        self.history_mask = np.zeros(self.volume_shape, dtype=np.float32)
        self.entropy_map = self._calculate_entropy_map()
        self.steps_taken = 0
        
        # 避免mask全为0，随机激活一个点
        if np.sum(self.current_mask) == 0:
            idx = tuple(np.random.randint(0, s) for s in self.current_mask.shape)
            self.current_mask[idx] = 1
        
        # Get environment config
        env_config = config.get('env', {})
        region_config = env_config.get('region_growing', {})
        
        # Initialize region grower with config
        self.region_grower = RegionGrower(
            similarity_threshold=env_config.get('similarity_threshold', 0.3),
            min_size=env_config.get('min_region_size', 8),
            max_size=env_config.get('max_region_size', 2000),
            connectivity=env_config.get('connectivity', 26),
            modality_weights=region_config.get('modality_weights', [0.4, 0.3, 0.3])
        )
        self.connectivity = env_config.get('connectivity', 26)  # For connectivity analysis
        
        # Load UNet model if provided
        self.model = None
        if model_path:
            self.model = UNet3D(in_channels=3).to(device)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        
        # Initialize episode state
        self.current_volume = None
        self.cancer_mask = None  # Ground truth
        self.steps_taken = 0
        
        if use_prompt:
            self.prompt_segmentor = PromptSegmentation(
                distance_threshold=config.get('prompt_distance_threshold', 0.2),
                temperature=config.get('prompt_temperature', 0.1),
                min_prob=config.get('prompt_min_prob', 0.1),
                device=device
            )
        
    def _resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Resize volume to standard size"""
        # Resize each modality
        if len(volume.shape) == 4:  # (C, H, W, D)
            if volume.shape[1:] == STANDARD_SIZE:
                return volume
            
            # Calculate scale factors for spatial dimensions
            scale_factors = np.array(STANDARD_SIZE) / np.array(volume.shape[1:])
            resized = np.stack([
                zoom(volume[i], scale_factors, order=1)
                for i in range(volume.shape[0])
            ])
        else:  # (H, W, D)
            if volume.shape == STANDARD_SIZE:
                return volume
            
            # Calculate scale factors
            scale_factors = np.array(STANDARD_SIZE) / np.array(volume.shape)
            resized = zoom(volume, scale_factors, order=1)
            
        return resized
    
    def _calculate_entropy_map(self) -> np.ndarray:
        """Calculate entropy map using the model"""
        if self.model is None:
            return np.zeros(self.volume_shape)
            
        with torch.no_grad():
            volume_tensor = torch.FloatTensor(self.current_volume).unsqueeze(0)
            volume_tensor = volume_tensor.to(self.device)
            predictions = self.model(volume_tensor)
            predictions = torch.sigmoid(predictions).cpu().numpy()
            
            # Calculate entropy
            p = np.clip(predictions, 1e-10, 1-1e-10)
            entropy = -(p * np.log(p) + (1-p) * np.log(1-p))
            return entropy[0, 0]  # Remove batch and channel dims
    
    def _calculate_dice(self, pred_mask: np.ndarray, true_mask: np.ndarray) -> float:

        intersection = np.sum(pred_mask * true_mask)
        union = np.sum(pred_mask) + np.sum(true_mask)
        if union == 0:
            return 0.0
        return 2.0 * intersection / union

    def _calculate_reward(self, new_region_mask: np.ndarray) -> float:

        try:
            # Calculate base metrics
            old_dice = self._calculate_dice(self.current_mask, self.cancer_mask)
            new_mask = np.logical_or(self.current_mask, new_region_mask)
            new_dice = self._calculate_dice(new_mask, self.cancer_mask)
            
            old_iou = self._calculate_iou(self.current_mask, self.cancer_mask)
            new_iou = self._calculate_iou(new_mask, self.cancer_mask)
            
            # Calculate improvement
            iou_improvement = new_iou - old_iou
            dice_improvement = new_dice - old_dice
            
            # Calculate exploration factor
            exploration_factor = np.sum(self.history_mask) / np.prod(self.volume_shape)
            
            # Calculate entropy reward
            entropy_reward = 0.0
            if self.entropy_map is not None and np.any(new_region_mask):
                entropy_reward = np.mean(self.entropy_map[new_region_mask])
            
            # Use AdaptiveRewardShaper to calculate reward
            reward = self.reward_shaper.shape_reward(
                base_reward=dice_improvement,  # Use dice improvement as base reward
                iou=new_iou,
                entropy=entropy_reward,
                exploration_factor=exploration_factor,
                steps_taken=self.steps_taken,
                is_boundary=self._is_boundary_region(new_region_mask)
            )
            
            # Update history
            self.reward_shaper.update_history(
                reward=reward,
                iou=new_iou,
                episode_length=self.steps_taken
            )
            
            return float(reward)
            
        except Exception as e:
            print(f"Error in reward calculation: {str(e)}")
            return -0.1  # Return a small negative reward as error handling
            
    def _is_boundary_region(self, mask: np.ndarray) -> bool:

        from scipy.ndimage import binary_erosion
        boundary = binary_dilation(mask, iterations=1) & ~binary_erosion(mask, iterations=1)
        return np.any(boundary)

    def _get_random_point_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int]:

        if mask is None or mask.sum() == 0:
            # If it's a healthy patient, select a point from the high entropy region
            if self.entropy_map is not None:
                high_entropy_points = np.where(self.entropy_map > np.percentile(self.entropy_map, 90))
                if len(high_entropy_points[0]) > 0:
                    idx = np.random.randint(0, len(high_entropy_points[0]))
                    return (high_entropy_points[0][idx],
                           high_entropy_points[1][idx],
                           high_entropy_points[2][idx])
            return tuple(np.random.randint(0, s) for s in self.volume_shape)
        
        # Prioritize selecting points from the boundary region
        from scipy.ndimage import binary_dilation, binary_erosion
        boundary = binary_dilation(mask, iterations=2) & ~binary_erosion(mask, iterations=2)
        boundary_points = np.nonzero(boundary)
        
        if len(boundary_points[0]) > 0:
            idx = np.random.randint(0, len(boundary_points[0]))
            return (boundary_points[0][idx],
                   boundary_points[1][idx],
                   boundary_points[2][idx])
        else:
            nonzero = np.nonzero(mask)
            idx = np.random.randint(0, len(nonzero[0]))
            return (nonzero[0][idx], nonzero[1][idx], nonzero[2][idx])
    
    def _get_largest_connected_component(self, mask: np.ndarray) -> np.ndarray:
        """Get the largest connected component"""
        structure = generate_binary_structure(3, self.connectivity)
        labeled, num_features = label(mask, structure)
        if num_features == 0:
            return mask
        
        # Find the largest connected component
        sizes = np.bincount(labeled.ravel())[1:]
        largest_label = sizes.argmax() + 1
        return (labeled == largest_label).astype(np.float32)
        
    def _get_optimal_prompt_point(self, gt_mask: np.ndarray) -> Tuple[int, int, int]:
        """Select the optimal prompt point"""
        # 1. Calculate distance transform
        from scipy.ndimage import distance_transform_edt
        dist_transform = distance_transform_edt(gt_mask)
        
        # 2. Find the point farthest from the boundary (region center)
        center_points = np.where(dist_transform == np.max(dist_transform))
        center_point = (
            center_points[0][0], 
            center_points[1][0], 
            center_points[2][0]
        )
        
        # 3. Sample points in the center region
        radius = 3
        valid_points = []
        for z in range(max(0, center_point[0]-radius), 
                      min(gt_mask.shape[0], center_point[0]+radius+1)):
            for y in range(max(0, center_point[1]-radius), 
                         min(gt_mask.shape[1], center_point[1]+radius+1)):
                for x in range(max(0, center_point[2]-radius), 
                             min(gt_mask.shape[2], center_point[2]+radius+1)):
                    if gt_mask[z,y,x]:
                        valid_points.append((z,y,x))
        
        # 4. Select the best point
        if valid_points:
            return random.choice(valid_points)
        return center_point

    def _calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate IoU"""
        intersection = np.float32(np.sum(pred_mask * gt_mask))
        union = np.float32(np.sum(pred_mask) + np.sum(gt_mask) - intersection)
        iou = intersection / (union + 1e-10)
        return float(iou)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
        specific_case: Optional[str] = None
    ) -> Tuple[Dict, Dict]:
        """Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Additional options
            specific_case: If provided, load this specific patient case
        
        Returns:
            Observation and info
        """
        super().reset(seed=seed)
        
        if specific_case is not None:
            # Load a specific patient case instead of a random one
            self.current_volume, self.cancer_mask, self.current_idx, self.original_volume, self.case_id = self.dataset.get_specific_patient(specific_case)
        else:
            # Get a random patient as before
            self.current_volume, self.cancer_mask, self.current_idx, self.original_volume, self.case_id = self.dataset.get_random_patient()
            
        self.current_volume = self._resize_volume(self.current_volume)
        
        # 处理原始体积数据
        if self.original_volume is not None:
            self.original_volume = self._resize_volume(self.original_volume)
        
        if self.cancer_mask is not None:
            self.cancer_mask = self._resize_volume(self.cancer_mask)
        else:
            self.cancer_mask = np.zeros(STANDARD_SIZE)
            
        self.current_mask = np.zeros(self.volume_shape, dtype=np.float32)
        self.history_mask = np.zeros(self.volume_shape, dtype=np.float32)
        self.entropy_map = self._calculate_entropy_map()
        self.steps_taken = 0
        
        # Reset Dice related metrics
        self.current_dice = 0.0
        self.episode_dice_history = []
        self.dice_history = []
        
        # Reset IoU related metrics
        self.episode_iou_history = []
        
        if self.use_prompt:
            prompt_point = self._get_optimal_prompt_point(self.cancer_mask)
            self.history_mask[prompt_point] = 1
            
            grown_region, initial_score = self.region_grower.grow_region(
                volume=self.current_volume,
                seed_point=prompt_point,
                entropy_map=self.entropy_map
            )
            
            if np.sum(grown_region) > 0:
                self.current_mask = grown_region
                initial_iou = self._calculate_iou(self.current_mask, self.cancer_mask)
                initial_dice = self._calculate_dice(self.current_mask, self.cancer_mask)
                
                self.episode_iou_history.append(initial_iou)
                self.episode_dice_history.append(initial_dice)
                
                if initial_iou > self.best_iou:
                    self.best_iou = initial_iou
                if initial_dice > self.best_dice:
                    self.best_dice = initial_dice
        
        # Reset metrics
        self.raw_metrics = {'dice': 0.0, 'iou': 0.0}
        self.post_metrics = {'dice': 0.0, 'iou': 0.0}
        self.boundary_metrics = {}
        
        observation = {
            'volume': self.current_volume,
            'current_mask': self.current_mask,
            'entropy_map': self.entropy_map,
            'history_mask': self.history_mask
        }
        
        return observation, {}
        
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take a step in the environment"""
        try:
            # 1. Action preprocessing and boundary check
            action = np.clip(action, 0, np.array(self.volume_shape) - 1)
            seed_point = tuple(map(int, np.round(action)))
            
            # 2. Check if the seed point is valid
            if self.history_mask[seed_point]:
                current_dice = self._calculate_dice(self.current_mask, self.cancer_mask)
                return (
                    {
                        'volume': self.current_volume,
                        'current_mask': self.current_mask,
                        'entropy_map': self.entropy_map,
                        'history_mask': self.history_mask
                    },
                    -0.5,
                    False,
                    False,
                    {
                        'region_score': 0.0,
                        'steps_taken': self.steps_taken,
                        'current_iou': self._calculate_iou(self.current_mask, self.cancer_mask),
                        'best_iou': self.best_iou,
                        'episode_iou_history': self.episode_iou_history,
                        'invalid_action': True,
                        'dice': current_dice,
                        'avg_dice': np.mean(self.dice_history) if self.dice_history else 0.0,
                        'best_dice': self.best_dice
                    }
                )
            
            # 3. Region growing (add timeout mechanism)
            try:
                def timeout_handler(signum, frame):
                    raise TimeoutError("Region growing timeout")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                
                region_mask, region_score = self.region_grower.grow_region(
                    self.current_volume,
                    seed_point,
                    self.entropy_map
                )
                
                signal.alarm(0)
                
            except TimeoutError:
                region_mask = np.zeros_like(self.current_mask)
                region_mask[seed_point] = 1
                region_mask = binary_dilation(region_mask, iterations=3)
                region_score = 0.1
                
            except Exception as e:
                region_mask = np.zeros_like(self.current_mask)
                region_score = 0.0
            
            # 4. Calculate reward
            reward = self._calculate_reward(region_mask)
            
            # 5. Update state
            if np.sum(region_mask) > 0:
                self.current_mask = np.logical_or(self.current_mask, region_mask)
                self.history_mask = np.logical_or(self.history_mask, region_mask)
            
            # 6. Update steps and metrics
            self.steps_taken += 1
            self.current_iou = self._calculate_iou(self.current_mask, self.cancer_mask)
            self.current_dice = self._calculate_dice(self.current_mask, self.cancer_mask)
            
            self.episode_iou_history.append(self.current_iou)
            self.episode_dice_history.append(self.current_dice)
            
            # Update dice history record list, and limit maximum length to 1000, to prevent memory overflow
            self.dice_history.append(self.current_dice)
            if len(self.dice_history) > 1000:
                self.dice_history = self.dice_history[-1000:]
            
            if self.current_iou > self.best_iou:
                self.best_iou = self.current_iou
            if self.current_dice > self.best_dice:
                self.best_dice = self.current_dice
            
            # 7. Check termination condition
            terminated = False
            truncated = self.steps_taken >= self.max_steps
            
            # 8. Prepare observation and info
            observation = {
                'volume': self.current_volume,
                'current_mask': self.current_mask,
                'entropy_map': self.entropy_map,
                'history_mask': self.history_mask
            }
            
            info = {
                'current_iou': self.current_iou,
                'current_dice': self.current_dice,
                'best_iou': self.best_iou,
                'best_dice': self.best_dice,
                'steps_taken': self.steps_taken
            }
            
            # Apply post-processing at episode end
            if terminated or truncated:
                try:
                    # Prepare input data
                    current_volume_slice = np.transpose(self.current_volume, (1, 2, 3, 0))  # (D, H, W, C)
                    current_mask_3d = self.current_mask.astype(np.float32)
                    
                    # Apply post-processing
                    processed_mask = self.post_processor(
                        pred=current_mask_3d,
                        image=current_volume_slice,
                        threshold=0.5
                    )
                    
                    # Update metrics
                    self.raw_metrics = {
                        'iou': self.current_iou,
                        'dice': self.current_dice
                    }
                    
                    post_iou = self._calculate_iou(processed_mask, self.cancer_mask)
                    post_dice = self._calculate_dice(processed_mask, self.cancer_mask)
                    
                    self.post_metrics = {
                        'iou': post_iou,
                        'dice': post_dice
                    }
                    
                    # If post-processing is better, update best metrics
                    if post_iou > self.best_iou:
                        self.best_iou = post_iou
                    if post_dice > self.best_dice:
                        self.best_dice = post_dice
                    
                    # Calculate boundary metrics
                    self.boundary_metrics = self.post_processor.calculate_boundary_metrics(
                        processed_mask,
                        self.cancer_mask
                    )
                    
                except Exception as e:
                    print(f"Post-processing failed: {str(e)}")
                    self.post_metrics = self.raw_metrics.copy()
                    self.boundary_metrics = {
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0
                    }
                
                # Update info dictionary
                info.update({
                    'raw_metrics': self.raw_metrics,
                    'post_metrics': self.post_metrics,
                    'boundary_metrics': self.boundary_metrics,
                    'episode_iou_history': self.episode_iou_history,
                    'episode_dice_history': self.episode_dice_history
                })
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error in step: {str(e)}")
            
            # Ensure current metrics are calculated and returned in case of exception
            try:
                self.current_iou = self._calculate_iou(self.current_mask, self.cancer_mask)
                self.current_dice = self._calculate_dice(self.current_mask, self.cancer_mask)
            except:
                self.current_iou = 0.0
                self.current_dice = 0.0
            
            error_info = {
                'error': str(e),
                'steps_taken': self.steps_taken,
                'current_iou': self.current_iou,
                'current_dice': self.current_dice,
                'best_iou': self.best_iou,
                'best_dice': self.best_dice,
                'raw_metrics': {'iou': self.current_iou, 'dice': self.current_dice},
                'post_metrics': {'iou': self.current_iou, 'dice': self.current_dice},
                'boundary_metrics': {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                },
                'episode_iou_history': self.episode_iou_history,
                'episode_dice_history': self.episode_dice_history
            }
            
            observation = {
                'volume': self.current_volume,
                'current_mask': self.current_mask,
                'entropy_map': self.entropy_map,
                'history_mask': self.history_mask
            }
            
            return observation, -1.0, True, False, error_info

    def _get_observation(self) -> Dict:
        """Get current observation"""
        observation = super()._get_observation()
        
        # If prompt is enabled, add prompt related info
        if self.use_prompt and hasattr(self, 'prompt_mask'):
            observation['prompt_mask'] = self.prompt_mask
            
        return observation 