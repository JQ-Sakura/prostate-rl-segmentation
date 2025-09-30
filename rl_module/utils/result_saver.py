import os
import numpy as np
import nibabel as nib
import torch
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import heapq
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class SegmentationResult:
    """存储分割结果的数据类"""
    episode: int
    step: int
    dice_score: float
    iou_score: float
    case_id: str
    patient_id: str
    timestamp: str
    
    def __lt__(self, other):
        """用于优先队列比较，按dice_score排序"""
        return self.dice_score < other.dice_score

class TopResultsSaver:
    """保存训练中最好的K个分割结果"""
    
    def __init__(self, save_dir: str, top_k: int = 10):
        """
        初始化结果保存器
        
        Args:
            save_dir: 保存目录
            top_k: 保存最好的K个结果
        """
        self.save_dir = Path(save_dir)
        self.top_k = top_k
        self.results_heap = []  # 最小堆，存储top_k个结果
        
        # 创建保存目录结构
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "predictions").mkdir(exist_ok=True)
        (self.save_dir / "ground_truth").mkdir(exist_ok=True)
        (self.save_dir / "volumes").mkdir(exist_ok=True)
        (self.save_dir / "metadata").mkdir(exist_ok=True)
        
        # 加载已有结果（如果存在）
        self._load_existing_results()
        
        # 如果没有元数据文件，创建一个空的
        metadata_file = self.save_dir / "top_results_metadata.json"
        if not metadata_file.exists():
            self._save_metadata()
    
    def _load_existing_results(self):
        """加载已存在的结果记录"""
        metadata_file = self.save_dir / "top_results_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                for item in data.get('results', []):
                    result = SegmentationResult(**item)
                    heapq.heappush(self.results_heap, result)
            print(f"Loaded {len(self.results_heap)} existing results")
    
    def _save_metadata(self):
        """保存元数据到JSON文件"""
        metadata = {
            'top_k': self.top_k,
            'total_results': len(self.results_heap),
            'last_updated': datetime.now().isoformat(),
            'results': [asdict(result) for result in sorted(self.results_heap, reverse=True)]
        }
        
        metadata_file = self.save_dir / "top_results_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_case_info_from_env(self, env) -> Tuple[str, str]:
        """从环境中获取当前case的信息"""
        try:
            # 优先从env直接获取case_id
            if hasattr(env, 'case_id') and env.case_id:
                case_id = env.case_id
                if hasattr(env.dataset, 'samples') and env.dataset.samples:
                    # PICAI数据集模式 - 从case_id提取patient_id
                    for sample in env.dataset.samples:
                        if sample['case_id'] == case_id:
                            return case_id, sample['patient_id']
                    # 如果找不到，从case_id推断patient_id
                    patient_id = case_id.split('_')[0] if '_' in case_id else case_id
                    return case_id, patient_id
                else:
                    return case_id, case_id
            
            # 备用方案：从数据集获取当前case信息
            if hasattr(env.dataset, 'samples') and env.dataset.samples:
                # PICAI数据集模式
                current_idx = getattr(env, 'current_idx', 0)
                if current_idx < len(env.dataset.samples):
                    sample = env.dataset.samples[current_idx]
                    return sample['case_id'], sample['patient_id']
            elif hasattr(env.dataset, 'patient_dirs'):
                # 原始数据集模式
                current_idx = getattr(env, 'current_idx', 0)
                if current_idx < len(env.dataset.patient_dirs):
                    patient_id = env.dataset.patient_dirs[current_idx]
                    return patient_id, patient_id
            
            # 最后备用方案：使用时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"unknown_case_{timestamp}", f"unknown_patient_{timestamp}"
            
        except Exception as e:
            print(f"Warning: Could not get case info from environment: {e}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"unknown_case_{timestamp}", f"unknown_patient_{timestamp}"
    
    def should_save(self, dice_score: float) -> bool:
        """判断是否应该保存当前结果"""
        if len(self.results_heap) < self.top_k:
            return True
        
        # 如果当前结果比堆中最差的结果好，则应该保存
        worst_result = self.results_heap[0]
        return dice_score > worst_result.dice_score
    
    def save_result(self, 
                   env, 
                   episode: int, 
                   step: int, 
                   dice_score: float, 
                   iou_score: float,
                   pred_mask: np.ndarray,
                   gt_mask: np.ndarray,
                   volume: np.ndarray = None):
        """
        保存一个分割结果
        
        Args:
            env: 环境对象
            episode: 当前episode
            step: 当前step
            dice_score: Dice分数
            iou_score: IoU分数
            pred_mask: 预测掩码
            gt_mask: 真实掩码
            volume: 原始体数据（3个模态），如果为None则从env获取
        """
        if not self.should_save(dice_score):
            return False
        
        # 获取case信息
        case_id, patient_id = self._get_case_info_from_env(env)
        
        # 获取原始体积数据（用于可视化）
        if volume is None:
            # 优先使用原始体积数据，如果没有则使用当前体积数据
            if hasattr(env, 'original_volume') and env.original_volume is not None:
                volume = env.original_volume
            else:
                volume = env.current_volume
        
        # 创建结果记录
        result = SegmentationResult(
            episode=episode,
            step=step,
            dice_score=dice_score,
            iou_score=iou_score,
            case_id=case_id,
            patient_id=patient_id,
            timestamp=datetime.now().isoformat()
        )
        
        # 生成唯一的文件名
        filename_prefix = f"ep{episode}_st{step}_{case_id}_dice{dice_score:.4f}"
        
        try:
            # 保存预测掩码
            pred_mask_path = self.save_dir / "predictions" / f"{filename_prefix}_pred.nii.gz"
            pred_nii = nib.Nifti1Image(pred_mask.astype(np.float32), np.eye(4))
            nib.save(pred_nii, pred_mask_path)
            
            # 保存真实掩码
            gt_mask_path = self.save_dir / "ground_truth" / f"{filename_prefix}_gt.nii.gz"
            gt_nii = nib.Nifti1Image(gt_mask.astype(np.float32), np.eye(4))
            nib.save(gt_nii, gt_mask_path)
            
            # 保存三模态体数据
            # 分别保存三个模态
            modality_names = ['adc', 'hbv', 't2w']  # 根据实际模态调整
            for i, modality_name in enumerate(modality_names):
                if i < volume.shape[0]:  # 确保有足够的模态
                    volume_path = self.save_dir / "volumes" / f"{filename_prefix}_{modality_name}.nii.gz"
                    volume_nii = nib.Nifti1Image(volume[i].astype(np.float32), np.eye(4))
                    nib.save(volume_nii, volume_path)
            
            # 保存详细元数据
            detailed_metadata = {
                'episode': episode,
                'step': step,
                'dice_score': dice_score,
                'iou_score': iou_score,
                'case_id': case_id,
                'patient_id': patient_id,
                'timestamp': result.timestamp,
                'files': {
                    'prediction': str(pred_mask_path.name),
                    'ground_truth': str(gt_mask_path.name),
                    'volumes': [f"{filename_prefix}_{name}.nii.gz" for name in modality_names]
                },
                'statistics': {
                    'pred_mask_stats': {
                        'mean': float(np.mean(pred_mask)),
                        'std': float(np.std(pred_mask)),
                        'min': float(np.min(pred_mask)),
                        'max': float(np.max(pred_mask)),
                        'sum': float(np.sum(pred_mask))
                    },
                    'gt_mask_stats': {
                        'mean': float(np.mean(gt_mask)),
                        'std': float(np.std(gt_mask)),
                        'min': float(np.min(gt_mask)),
                        'max': float(np.max(gt_mask)),
                        'sum': float(np.sum(gt_mask))
                    }
                }
            }
            
            metadata_path = self.save_dir / "metadata" / f"{filename_prefix}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(detailed_metadata, f, indent=2)
            
            # 更新结果堆
            if len(self.results_heap) >= self.top_k:
                # 移除最差的结果
                worst_result = heapq.heappop(self.results_heap)
                self._remove_result_files(worst_result)
            
            # 添加新结果
            heapq.heappush(self.results_heap, result)
            
            # 更新元数据文件
            self._save_metadata()
            
            print(f"✅ Saved result: Episode {episode}, Step {step}, Dice: {dice_score:.4f}, IoU: {iou_score:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving result: {e}")
            return False
    
    def _remove_result_files(self, result: SegmentationResult):
        """删除一个结果的相关文件"""
        try:
            filename_prefix = f"ep{result.episode}_st{result.step}_{result.case_id}_dice{result.dice_score:.4f}"
            
            # 删除预测掩码
            pred_file = self.save_dir / "predictions" / f"{filename_prefix}_pred.nii.gz"
            if pred_file.exists():
                pred_file.unlink()
            
            # 删除真实掩码
            gt_file = self.save_dir / "ground_truth" / f"{filename_prefix}_gt.nii.gz"
            if gt_file.exists():
                gt_file.unlink()
            
            # 删除体数据文件
            modality_names = ['adc', 'hbv', 't2w']
            for modality_name in modality_names:
                volume_file = self.save_dir / "volumes" / f"{filename_prefix}_{modality_name}.nii.gz"
                if volume_file.exists():
                    volume_file.unlink()
            
            # 删除元数据文件
            metadata_file = self.save_dir / "metadata" / f"{filename_prefix}_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()
                
            print(f"🗑️  Removed files for result: Episode {result.episode}, Dice: {result.dice_score:.4f}")
            
        except Exception as e:
            print(f"Warning: Could not remove files for result {result.episode}: {e}")
    
    def get_top_results(self) -> List[SegmentationResult]:
        """获取当前保存的最好结果列表"""
        return sorted(self.results_heap, key=lambda x: x.dice_score, reverse=True)
    
    def print_summary(self):
        """打印当前保存结果的摘要"""
        if not self.results_heap:
            print("No results saved yet.")
            return
        
        print(f"\n📊 Top {len(self.results_heap)} Segmentation Results:")
        print("=" * 80)
        print(f"{'Rank':<5} {'Episode':<8} {'Step':<6} {'Dice':<8} {'IoU':<8} {'Case ID':<15} {'Timestamp'}")
        print("-" * 80)
        
        for i, result in enumerate(self.get_top_results(), 1):
            print(f"{i:<5} {result.episode:<8} {result.step:<6} {result.dice_score:<8.4f} "
                  f"{result.iou_score:<8.4f} {result.case_id:<15} {result.timestamp[:19]}")
        
        print("=" * 80)
        print(f"Best Dice Score: {max(r.dice_score for r in self.results_heap):.4f}")
        print(f"Average Dice Score: {sum(r.dice_score for r in self.results_heap) / len(self.results_heap):.4f}")
        print(f"Results saved in: {self.save_dir}") 