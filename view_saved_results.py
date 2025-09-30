#!/usr/bin/env python3
"""
查看和分析保存的训练结果的脚本
"""

import os
import json
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

class ResultsViewer:
    """结果查看器类"""
    
    def __init__(self, results_dir: str):
        """
        初始化结果查看器
        
        Args:
            results_dir: 结果保存目录
        """
        self.results_dir = Path(results_dir)
        self.metadata_file = self.results_dir / "top_results_metadata.json"
        
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
    
    def load_metadata(self) -> Dict[str, Any]:
        """加载元数据"""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def print_summary(self):
        """打印结果摘要"""
        metadata = self.load_metadata()
        results = metadata.get('results', [])
        
        if not results:
            print("No results found.")
            return
        
        print(f"\n📊 Top {len(results)} Segmentation Results Summary")
        print("=" * 80)
        print(f"Results Directory: {self.results_dir}")
        print(f"Last Updated: {metadata.get('last_updated', 'Unknown')}")
        print(f"Total Saved Results: {metadata.get('total_results', 0)}")
        print("-" * 80)
        
        print(f"{'Rank':<5} {'Episode':<8} {'Step':<6} {'Dice':<8} {'IoU':<8} {'Case ID':<15} {'Patient ID':<15}")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"{i:<5} {result['episode']:<8} {result['step']:<6} "
                  f"{result['dice_score']:<8.4f} {result['iou_score']:<8.4f} "
                  f"{result['case_id']:<15} {result['patient_id']:<15}")
        
        # 计算统计信息
        dice_scores = [r['dice_score'] for r in results]
        iou_scores = [r['iou_score'] for r in results]
        
        print("=" * 80)
        print("📈 Statistics:")
        print(f"Best Dice Score: {max(dice_scores):.4f}")
        print(f"Average Dice Score: {np.mean(dice_scores):.4f}")
        print(f"Dice Score Std: {np.std(dice_scores):.4f}")
        print(f"Best IoU Score: {max(iou_scores):.4f}")
        print(f"Average IoU Score: {np.mean(iou_scores):.4f}")
        print(f"IoU Score Std: {np.std(iou_scores):.4f}")
        print("=" * 80)
    
    def get_result_files(self, rank: int) -> Dict[str, Path]:
        """
        获取指定排名结果的文件路径
        
        Args:
            rank: 排名（1-based）
            
        Returns:
            包含文件路径的字典
        """
        metadata = self.load_metadata()
        results = metadata.get('results', [])
        
        if rank < 1 or rank > len(results):
            raise ValueError(f"Rank {rank} is out of range (1-{len(results)})")
        
        result = results[rank - 1]  # Convert to 0-based index
        
        # 构建文件名前缀
        filename_prefix = f"ep{result['episode']}_st{result['step']}_{result['case_id']}_dice{result['dice_score']:.4f}"
        
        files = {
            'prediction': self.results_dir / "predictions" / f"{filename_prefix}_pred.nii.gz",
            'ground_truth': self.results_dir / "ground_truth" / f"{filename_prefix}_gt.nii.gz",
            'metadata': self.results_dir / "metadata" / f"{filename_prefix}_metadata.json",
            'volumes': {}
        }
        
        # 添加体数据文件
        modality_names = ['adc', 'hbv', 't2w']
        for modality in modality_names:
            volume_file = self.results_dir / "volumes" / f"{filename_prefix}_{modality}.nii.gz"
            if volume_file.exists():
                files['volumes'][modality] = volume_file
        
        return files
    
    def load_result_data(self, rank: int) -> Dict[str, Any]:
        """
        加载指定排名结果的数据
        
        Args:
            rank: 排名（1-based）
            
        Returns:
            包含所有数据的字典
        """
        files = self.get_result_files(rank)
        
        data = {
            'prediction': None,
            'ground_truth': None,
            'volumes': {},
            'metadata': None
        }
        
        # 加载预测掩码
        if files['prediction'].exists():
            data['prediction'] = nib.load(files['prediction']).get_fdata()
        
        # 加载真实掩码
        if files['ground_truth'].exists():
            data['ground_truth'] = nib.load(files['ground_truth']).get_fdata()
        
        # 加载体数据
        for modality, file_path in files['volumes'].items():
            if file_path.exists():
                data['volumes'][modality] = nib.load(file_path).get_fdata()
        
        # 加载详细元数据
        if files['metadata'].exists():
            with open(files['metadata'], 'r') as f:
                data['metadata'] = json.load(f)
        
        return data
    
    def visualize_result(self, rank: int, slice_idx: int = None, save_path: str = None):
        """
        可视化指定排名的结果
        
        Args:
            rank: 排名（1-based）
            slice_idx: 切片索引，如果为None则自动选择有病灶的切片
            save_path: 保存图像的路径，如果为None则显示图像
        """
        data = self.load_result_data(rank)
        
        pred_mask = data['prediction']
        gt_mask = data['ground_truth']
        volumes = data['volumes']
        
        if pred_mask is None or gt_mask is None:
            print(f"Error: Could not load data for rank {rank}")
            return
        
        # 智能选择切片
        if slice_idx is None:
            # 找到有预测结果的切片
            pred_slices = []
            gt_slices = []
            
            for i in range(pred_mask.shape[2]):
                pred_count = np.count_nonzero(pred_mask[:,:,i])
                gt_count = np.count_nonzero(gt_mask[:,:,i])
                if pred_count > 0:
                    pred_slices.append((i, pred_count))
                if gt_count > 0:
                    gt_slices.append((i, gt_count))
            
            # 优先选择预测和真实都有病灶的切片
            if pred_slices and gt_slices:
                # 找到预测和真实都有病灶的切片
                pred_slice_indices = set([s[0] for s in pred_slices])
                gt_slice_indices = set([s[0] for s in gt_slices])
                common_slices = pred_slice_indices.intersection(gt_slice_indices)
                
                if common_slices:
                    # 选择中间的共同切片
                    common_slices = sorted(list(common_slices))
                    slice_idx = common_slices[len(common_slices)//2]
                    print(f"自动选择切片 {slice_idx} (预测和真实都有病灶)")
                elif pred_slices:
                    # 如果没有共同切片，选择预测结果最多的切片
                    slice_idx = max(pred_slices, key=lambda x: x[1])[0]
                    print(f"自动选择切片 {slice_idx} (有预测结果)")
                else:
                    # 如果预测没有结果，选择真实病灶最多的切片
                    slice_idx = max(gt_slices, key=lambda x: x[1])[0]
                    print(f"自动选择切片 {slice_idx} (有真实病灶)")
            else:
                # 如果都没有，使用默认中间切片
                slice_idx = pred_mask.shape[2] // 2
                print(f"未发现病灶，使用默认中间切片 {slice_idx}")
            
            # 打印切片信息
            pred_count = np.count_nonzero(pred_mask[:,:,slice_idx])
            gt_count = np.count_nonzero(gt_mask[:,:,slice_idx])
            print(f"选择的切片 {slice_idx}: 预测像素={pred_count}, 真实像素={gt_count}")
        else:
            # 检查用户指定切片的信息
            pred_count = np.count_nonzero(pred_mask[:,:,slice_idx])
            gt_count = np.count_nonzero(gt_mask[:,:,slice_idx])
            print(f"用户指定切片 {slice_idx}: 预测像素={pred_count}, 真实像素={gt_count}")
            if pred_count == 0 and gt_count == 0:
                print(f"警告: 切片 {slice_idx} 没有任何病灶！")
        
        # 创建图像
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 显示三个模态的体数据
        modality_names = ['adc', 'hbv', 't2w']
        for i, modality in enumerate(modality_names):
            if modality in volumes:
                volume = volumes[modality]
                # 应用医学影像标准方向：左右翻转
                volume_slice = np.fliplr(volume[:, :, slice_idx])
                
                # 改进的显示方法：使用合适的对比度
                p5, p95 = np.percentile(volume_slice, [5, 95])
                if p95 > p5:
                    axes[0, i].imshow(volume_slice, cmap='gray', vmin=p5, vmax=p95)
                else:
                    axes[0, i].imshow(volume_slice, cmap='gray')
                axes[0, i].set_title(f'{modality.upper()} (Slice {slice_idx})')
                axes[0, i].axis('off')
            else:
                axes[0, i].text(0.5, 0.5, f'{modality.upper()}\nNot Available', 
                               ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].axis('off')
        
        # 显示预测掩码（应用相同的方向修正）
        pred_slice = np.fliplr(pred_mask[:, :, slice_idx])
        axes[1, 0].imshow(pred_slice, cmap='hot')
        axes[1, 0].set_title('Prediction Mask')
        axes[1, 0].axis('off')
        
        # 显示真实掩码（应用相同的方向修正）
        gt_slice = np.fliplr(gt_mask[:, :, slice_idx])
        axes[1, 1].imshow(gt_slice, cmap='hot')
        axes[1, 1].set_title('Ground Truth Mask')
        axes[1, 1].axis('off')
        
        # 显示重叠图像
        if 't2w' in volumes:
            # 应用相同的方向修正
            background = np.fliplr(volumes['t2w'][:, :, slice_idx])
            
            # 改进的背景图像归一化
            p5, p95 = np.percentile(background, [5, 95])
            if p95 > p5:
                background = np.clip(background, p5, p95)
                background = (background - p5) / (p95 - p5)
            else:
                background = (background - background.min()) / (background.max() - background.min() + 1e-8)
            
            # 创建重叠图像
            overlay = background.copy()
            overlay = np.stack([overlay, overlay, overlay], axis=-1)
            
            # 添加预测掩码（红色）- 使用已修正的切片
            overlay[:, :, 0] = np.where(pred_slice > 0.5, 1, overlay[:, :, 0])
            
            # 添加真实掩码轮廓（绿色）- 使用已修正的切片
            from scipy.ndimage import binary_erosion, binary_dilation
            gt_contour = binary_dilation(gt_slice > 0.5) & ~binary_erosion(gt_slice > 0.5)
            overlay[:, :, 1] = np.where(gt_contour, 1, overlay[:, :, 1])
            
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('Overlay (Red: Pred, Green: GT)')
            axes[1, 2].axis('off')
        else:
            axes[1, 2].text(0.5, 0.5, 'No T2W image\nfor overlay', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].axis('off')
        
        # 添加标题
        metadata = data.get('metadata', {})
        dice_score = metadata.get('dice_score', 0)
        iou_score = metadata.get('iou_score', 0)
        episode = metadata.get('episode', 0)
        step = metadata.get('step', 0)
        case_id = metadata.get('case_id', 'Unknown')
        
        fig.suptitle(f'Rank {rank} - Episode {episode}, Step {step}\n'
                    f'Case: {case_id}, Dice: {dice_score:.4f}, IoU: {iou_score:.4f}', 
                    fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Image saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="查看训练结果")
    parser.add_argument("results_dir", help="结果保存目录")
    parser.add_argument("--summary", action="store_true", help="显示结果摘要")
    parser.add_argument("--visualize", type=int, help="可视化指定排名的结果 (1-based)")
    parser.add_argument("--slice", type=int, help="指定要显示的切片索引")
    parser.add_argument("--save", type=str, help="保存可视化图像的路径")
    
    args = parser.parse_args()
    
    try:
        viewer = ResultsViewer(args.results_dir)
        
        if args.summary:
            viewer.print_summary()
        
        if args.visualize:
            viewer.visualize_result(args.visualize, args.slice, args.save)
        
        if not args.summary and not args.visualize:
            # 默认显示摘要
            viewer.print_summary()
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 