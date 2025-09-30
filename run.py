import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import yaml
import argparse
from rl_module.agents.ppo_agent import PPOAgent
from rl_module.env.cancer_env import CancerEnv
from rl_module.utils.adaptive_reward import AdaptiveRewardShaper
from rl_module.utils.result_saver import TopResultsSaver
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
import socket
import random
import tempfile
import shutil
import glob
import gc
from contextlib import nullcontext
import torch.nn as nn
from pathlib import Path
import nibabel as nib
from evaluation.metrics import calculate_metrics

# 配置文件路径
DEFAULT_CONFIG_PATH = 'rl_module/configs/train_config.yaml'

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {str(e)}")
        raise

def find_free_port():
    """Find a free port to use for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
        # Ensure the port is really released
        s.close()
        time.sleep(1)
    return port

def setup_distributed(rank, world_size, config):
    """Setup distributed training environment"""
    max_retries = 5
    retry_delay = 2
    
    for retry in range(max_retries):
        try:
            # Clean up the previous environment
            cleanup_distributed()
            
            # Ensure the CUDA device is correctly set
            if torch.cuda.is_available():
                torch.cuda.set_device(rank)
            
            # Dynamically find an available port
            if rank == 0:
                port = find_free_port()
                # Write the port number to the temporary file
                with open('.port_file', 'w') as f:
                    f.write(str(port))
            
            # Wait for the main process to write the port number
            if rank != 0:
                while not os.path.exists('.port_file'):
                    time.sleep(0.1)
                with open('.port_file', 'r') as f:
                    port = int(f.read().strip())
            
            # Set environment variables
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = str(port)
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['RANK'] = str(rank)
            
            # 增加NCCL通信超时设置
            os.environ['NCCL_BLOCKING_WAIT'] = '1'
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
            os.environ['NCCL_DEBUG'] = 'INFO'
            os.environ['NCCL_IB_TIMEOUT'] = '30'  # 增加通信超时时间
            
            if rank == 0:
                print(f"Attempting to initialize distributed training on port {port} (attempt {retry + 1}/{max_retries})")
            
            # Initialize the process group with更长的超时时间
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://127.0.0.1:{port}',
                world_size=world_size,
                rank=rank,
                timeout=timedelta(minutes=10)  # 增加超时时间到10分钟
            )
            
            # Ensure all processes are initialized
            dist.barrier()
            
            if rank == 0:
                print(f"Successfully initialized distributed training on port {port}")
                # Clean up the temporary file
                if os.path.exists('.port_file'):
                    os.remove('.port_file')
            return
            
        except Exception as e:
            if rank == 0:
                print(f"Attempt {retry + 1}/{max_retries} failed: {str(e)}")
                # Clean up the temporary file
                if os.path.exists('.port_file'):
                    os.remove('.port_file')
            
            # Clean up the environment
            cleanup_distributed()
            
            if retry == max_retries - 1:
                raise RuntimeError(f"Failed to initialize distributed training after {max_retries} attempts")
            
            # Wait before retrying
            time.sleep(retry_delay * (retry + 1))

def is_port_available(port):
    """Check if the port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return True
    except:
        return False

def cleanup_distributed():
    """Cleanup distributed training environment"""
    try:
        if dist.is_initialized():
            dist.barrier()  # Ensure all processes reach here
            dist.destroy_process_group()
    except Exception as e:
        print(f"Warning in cleanup_distributed: {str(e)}")
    finally:
        # Clean up environment variables
        for env_var in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK']:
            if env_var in os.environ:
                del os.environ[env_var]
        
        # Clean up temporary files
        if os.path.exists('.port_file'):
            try:
                os.remove('.port_file')
            except:
                pass
        
        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Reset CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device('cuda:0')

def train(config):
    """Training function for single GPU"""
    try:
        # 设置CUDA环境变量避免内存分配器问题
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA操作便于调试
        os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6'  # 限制支持的架构
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Update CUDA performance optimization configuration
        if torch.cuda.is_available():
            try:
                # Update CUDA memory allocator configuration - 禁用expandable_segments避免内存分配器断言失败
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8,backend:native'
                
                # Set memory management strategy - 使用更保守的内存设置
                torch.cuda.set_per_process_memory_fraction(0.7)  # 使用70%的GPU内存，更保守
                
                # 清理现有内存
                torch.cuda.empty_cache()
                gc.collect()
                
                # Set training parameters - 减少batch size避免内存问题
                config['training']['batch_size'] = 2  # 进一步减少batch size
                config['training']['ppo_epochs'] = 2
                config['training']['update_interval'] = 8  # 增加更新间隔
                config['training']['max_steps_per_episode'] = 32
                
                # 强制同步GPU操作
                torch.cuda.synchronize()
                
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB total")
                print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
                print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.1f}GB")
                
            except Exception as e:
                print(f"Warning: CUDA optimization setup failed: {e}")
                # 如果CUDA设置失败，使用最基本的配置
                config['training']['batch_size'] = 1
                config['training']['update_interval'] = 16
        
        # Enable CUDA graph optimization
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
            torch.cuda.set_sync_debug_mode(0)
            
            print("\n=== Training Configuration ===")
            print(f"Total Episodes: {config['training']['num_episodes']}")
            print(f"Steps per Episode: {config['training']['max_steps_per_episode']}")
            print(f"Update Interval: {config['training']['update_interval']}")
            print(f"Batch Size: {config['training']['batch_size']}")
            print(f"GPU Memory Limit: 80%")
            print(f"PyTorch Version: {torch.__version__}")
        
        # Initialize environment and agent
            env = CancerEnv(
            data_dir=config['paths']['data_dir'],
                config=config,
            model_path=config['paths']['model_path'],
            max_steps=config['training']['max_steps_per_episode'],
            device=device
        )
        
        agent = PPOAgent(
            state_dim=env.observation_space['volume'].shape,
            action_dim=3,
            device=device,
            lr=config['ppo']['learning_rate'],
            gamma=config['ppo']['gamma'],
            epsilon=config['ppo']['epsilon'],
            c1=config['ppo']['c1'],
            c2=config['ppo']['c2']
        )
        
        # Training loop
        best_reward = float('-inf')
        episode_rewards = []
        
        # 初始化结果保存器和最佳值跟踪
        results_save_dir = config.get('training', {}).get('results_save_dir', 'training_results')
        top_k = config.get('training', {}).get('save_top_k_results', 10)
        result_saver = TopResultsSaver(save_dir=results_save_dir, top_k=top_k)
        
        # 跟踪全局最佳值，只有超过时才保存
        global_best_dice = 0.0
        global_best_iou = 0.0
        save_threshold = config.get('training', {}).get('save_threshold_dice', 0.1)
        
        print(f"📁 Results will be saved to: {results_save_dir}")
        print(f"🏆 Saving top {top_k} segmentation results")
        print(f"⚡ Optimization: Only saving results when dice > current best ({global_best_dice:.4f}) and > threshold ({save_threshold})")
        
        if config.get('training', {}).get('use_wandb', False):
            wandb.init(
                project=config.get('wandb', {}).get('project', 'cancer-rl'),
                entity=config.get('wandb', {}).get('entity', None),
                config=config
            )
        
        for episode in range(config['training']['num_episodes']):
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done:
                action_raw, action_scaled, log_prob, value = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action_scaled)
                done = terminated or truncated
                agent.memory.push(state, action_raw, reward, value, log_prob, done)
                state = next_state
                episode_reward += reward
                step_count += 1
                
                # 在每一步后检查是否保存结果 - 优化版本：只有超过全局最佳值时才保存
                if 'current_dice' in info and 'current_iou' in info:
                    current_dice = info['current_dice']
                    current_iou = info['current_iou']
                    
                    # 添加调试信息（减少频率避免日志过多）
                    if step_count % 10 == 0:  # 每10步打印一次
                        print(f"🔍 Episode {episode}, Step {step_count}: Dice={current_dice:.4f} (best: {global_best_dice:.4f}), IoU={current_iou:.4f} (best: {global_best_iou:.4f})")
                    
                    # 优化逻辑：只有当前结果超过全局最佳值且超过阈值时才保存
                    is_new_best_dice = current_dice > global_best_dice
                    is_above_threshold = current_dice > save_threshold
                    
                    if is_new_best_dice and is_above_threshold:
                        try:
                            # 保存新的最佳结果
                            result_saver.save_result(
                                env=env,
                                episode=episode,
                                step=step_count,
                                dice_score=current_dice,
                                iou_score=current_iou,
                                pred_mask=env.current_mask,
                                gt_mask=env.cancer_mask
                                # volume参数移除，让方法自动从env获取原始体积数据
                            )
                            
                            # 更新全局最佳值
                            global_best_dice = current_dice
                            global_best_iou = max(global_best_iou, current_iou)
                            
                            print(f"🏆 NEW BEST! Saved result - Dice: {current_dice:.4f}, IoU: {current_iou:.4f}")
                            
                        except Exception as e:
                            print(f"⚠️  Warning: Failed to save result at episode {episode}, step {step_count}: {e}")
                    
                    elif not is_above_threshold and step_count % 20 == 0:  # 减少日志频率
                        print(f"📊 Dice {current_dice:.4f} ≤ threshold {save_threshold}, not saving")
                    elif not is_new_best_dice and current_dice > save_threshold and step_count % 20 == 0:
                        print(f"📊 Dice {current_dice:.4f} ≤ current best {global_best_dice:.4f}, not saving")
                
                if len(agent.memory) >= config['training']['update_interval']:
                    try:
                        agent.update()
                    except RuntimeError as e:
                        if "CUDA" in str(e) and "memory" in str(e).lower():
                            print(f"⚠️  CUDA memory error during update: {e}")
                            print("🧹 Clearing CUDA cache and retrying with smaller batch...")
                            
                            # 清理内存
                            torch.cuda.empty_cache()
                            gc.collect()
                            torch.cuda.synchronize()
                            
                            # 临时减少batch size重试
                            original_batch_size = config['training']['batch_size']
                            config['training']['batch_size'] = max(1, original_batch_size // 2)
                            
                            try:
                                agent.update()
                                print(f"✅ Update successful with reduced batch size: {config['training']['batch_size']}")
                                # 恢复原始batch size
                                config['training']['batch_size'] = original_batch_size
                            except Exception as retry_e:
                                print(f"❌ Retry failed: {retry_e}")
                                # 清空memory继续训练
                                agent.memory.clear()
                                config['training']['batch_size'] = 1  # 使用最小batch size
                        else:
                            raise e
            
            episode_rewards.append(episode_reward)
            
            # 定期内存清理避免内存碎片化
            if episode % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                if episode % 50 == 0:  # 每50个episode强制同步
                    torch.cuda.synchronize()
                    print(f"🧹 Memory cleanup - Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB, Cached: {torch.cuda.memory_reserved() / 1e9:.1f}GB")
            
            # Logging and saving
            if episode % config['training']['save_interval'] == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    agent.save(f"best_model_episode_{episode}.pth")
                
                # 每100个episode打印一次结果摘要和当前最佳值
                if episode % 100 == 0:
                    result_saver.print_summary()
                    print(f"🏆 Current global best - Dice: {global_best_dice:.4f}, IoU: {global_best_iou:.4f}")
            # wandb logging
            if config.get('training', {}).get('use_wandb', False):
                wandb.log({
                    'episode': episode,
                    'reward': episode_reward,
                    'best_reward': best_reward,
                    'current_iou': info.get('current_iou', 0),
                    'current_dice': info.get('current_dice', 0),
                    'global_best_dice': global_best_dice,  # 新增全局最佳dice跟踪
                    'global_best_iou': global_best_iou,    # 新增全局最佳iou跟踪
                    'best_iou': info.get('best_iou', 0),
                    'best_dice': getattr(env, 'best_dice', 0),
                    'policy_loss': getattr(agent, 'last_policy_loss', 0),
                    'value_loss': getattr(agent, 'last_value_loss', 0),
                    'entropy': getattr(agent, 'last_entropy', 0),
                    'saved_results_count': len(result_saver.top_results) if hasattr(result_saver, 'top_results') else 0,  # 跟踪保存的结果数量
                })
        
        if config.get('training', {}).get('use_wandb', False):
            wandb.finish()
        
        # 打印最终的结果摘要
        print("\n" + "="*80)
        print("🎉 Training completed! Final results summary:")
        result_saver.print_summary()
        print(f"🏆 Final global best values - Dice: {global_best_dice:.4f}, IoU: {global_best_iou:.4f}")
        print(f"💾 Total results saved: {len(result_saver.top_results) if hasattr(result_saver, 'top_results') else 'Unknown'}")
        print("="*80)
        
        return agent
                
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def validate(agent, val_env, device, config):
    """
    Validate model performance on validation dataset
    
    Args:
        agent: PPO agent
        val_env: Validation environment
        device: Computing device
        config: Configuration dictionary
        
    Returns:
        Dictionary of validation metrics
    """
    print("Starting validation...")
    
    # Set agent to evaluation mode
    agent.ac.eval()
    
    # Ensure environment is valid
    if val_env is None:
        print("Warning: Validation environment not available, skipping validation")
        return {"val_iou": 0.0, "val_dice": 0.0, "val_reward": 0.0}
    
    # Clean memory before validation
    torch.cuda.empty_cache()
    
    # Only run one validation episode to avoid getting stuck
    try:
        # Reset environment
        state, _ = val_env.reset()
        
        episode_reward = 0
        episode_steps = 0
        done = False
        truncated = False
        
        # Maximum steps limited to 8 to avoid validation taking too long
        max_steps = min(8, config.get('training', {}).get('max_steps_per_episode', 8))
        
        # Run a complete validation cycle
        while not (done or truncated) and episode_steps < max_steps:
            # Process input data
            with torch.no_grad():
                if 'volume' in state:
                    volume = torch.FloatTensor(state['volume']).unsqueeze(0)
                    
                    # Prepare masks
                    if 'masks' not in state:
                        masks = torch.FloatTensor(np.stack([
                            state['current_mask'],
                            state.get('entropy_map', np.zeros_like(state['current_mask'])),
                            state.get('history_mask', np.zeros_like(state['current_mask']))
                        ])).unsqueeze(0)
                    else:
                        masks = torch.FloatTensor(state['masks']).unsqueeze(0)
                    
                    # Move data to device
                    volume = volume.to(device)
                    masks = masks.to(device)
                    
                    # Model inference
                    state_dict = {
                        'volume': volume,
                        'masks': masks
                    }
                    
                    try:
                        output = agent.ac(state_dict)
                        if isinstance(output, tuple) and len(output) >= 2:
                            action_dist, _ = output[:2]
                            action = action_dist.mean
                        else:
                            print("Warning: Unexpected model output format")
                            break
                    except Exception as e:
                        print(f"Model inference failed: {str(e)}")
                        break
                        
                    # Execute action
                    try:
                        step_result = val_env.step(action.cpu().numpy()[0])
                        if isinstance(step_result, tuple) and len(step_result) >= 5:
                            next_state, reward, done, truncated, info = step_result
                        else:
                            print("Warning: Unexpected step result format")
                            break
                    except Exception as e:
                        print(f"Environment step execution failed: {str(e)}")
                        break
                    
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                else:
                    print("Warning: State does not contain volume field")
                    break
        
        # Get final metrics
        val_iou = info.get('current_iou', 0)
        val_dice = info.get('current_dice', 0)
        best_dice = info.get('best_dice', 0)
        
        print(f"Validation completed - Steps: {episode_steps}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        
        # Build results dictionary
        val_metrics = {
            "val_iou": val_iou,
            "val_dice": val_dice,
            "val_best_dice": best_dice,
            "val_reward": episode_reward,
            "val_steps": episode_steps
        }
        
        # Set agent back to training mode
        agent.ac.train()
        
        # Clean memory
        torch.cuda.empty_cache()
        
        return val_metrics
        
    except Exception as e:
        print(f"Error during validation process: {str(e)}")
        return {"val_iou": 0.0, "val_dice": 0.0, "val_reward": 0.0}

def validate_config(config):
    """Validate configuration and add missing default values"""
    default_config = {
        'paths': {
            'data_dir': '/raid/candi/junqing/FYP/image_with_masks',
            'log_dir': 'logs',
            'model_path': None
        },
        'env': {
            'similarity_threshold': 0.3,
            'min_region_size': 8,
            'max_region_size': 2000,
            'connectivity': 26,
            'use_prompt': True,
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
                'adaptation_rate': 0.01
            }
        },
        'training': {
            'num_episodes': 1000,
            'max_steps_per_episode': 32,
            'update_interval': 4,
            'ppo_epochs': 2,
            'batch_size': 4,
            'learning_rate': 0.0001,
            'min_lr': 1e-6,
            'use_wandb': True,
            'patience': 20,
            'lr_decay_factor': 0.5,
            'save_interval': 100,
            'use_single_gpu': False,  # 默认使用单GPU训练，避免分布式通信问题
            'results_save_dir': 'training_results',
            'save_top_k_results': 10
        },
        'ppo': {
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'epsilon': 0.2,
            'c1': 0.5,
            'c2': 0.01,
            'clip_grad_norm': 0.5,
            'gae_lambda': 0.95,
            'adaptive_lr': False
        },
        'validation': {
            'interval': 1,
            'validation_interval': 10,
            'metrics': ['dice', 'iou', 'precision', 'recall'],
            'best_metric': 'dice'
        },
        'wandb': {
            'project': 'cancer-rl',
            'entity': None,
            'tags': ['training'],
            'notes': 'Training with validation'
        },
        'data': {
            'split_ratio': {
                'train': 0.7,
                'valid': 0.15,
                'test': 0.15
            }
        }
    }
    
    # Recursively merge configurations
    def merge_configs(target, source):
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target[key], dict):
                merge_configs(target[key], value)
    
    merge_configs(config, default_config)
    print("Configuration validation complete, missing default values added")

    # === 新增：自动注入PICAI数据集路径 ===
    config['data']['picai_folds'] = [
        '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold0',
        '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold1',
        '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold2',
        '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold3',
        '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold4',
    ]
    config['data']['mask_dir'] = '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_labels-main/csPCa_lesion_delineations/human_expert/original'
    config['data']['keep_cases_path'] = '/raid/candi/junqing/FYP-SERVER/rl_module/data/keep_cases.txt'
    # === END ===

    return config

def main():
    """Main function"""
    try:
        # Load configuration
        config = load_config(DEFAULT_CONFIG_PATH)
        
        # Validate configuration
        config = validate_config(config)
        
        # === 新增：自动注入PICAI数据集路径 ===
        config['data']['picai_folds'] = [
            '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold0',
            '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold1',
            '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold2',
            '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold3',
            '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold4',
        ]
        config['data']['mask_dir'] = '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_labels-main/csPCa_lesion_delineations/human_expert/original'
        config['data']['keep_cases_path'] = '/raid/candi/junqing/FYP-SERVER/rl_module/data/keep_cases.txt'
        
        # Set single GPU training
        config['training']['use_single_gpu'] = True
        
        # Start training
        agent = train(config)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\n[Error] {str(e)}")
        raise

if __name__ == '__main__':
    main() 