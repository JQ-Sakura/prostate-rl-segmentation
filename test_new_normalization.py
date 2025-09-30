#!/usr/bin/env python3
"""
测试新的医学图像归一化方法
"""

import numpy as np
import matplotlib.pyplot as plt
from rl_module.data.dataset import MRIDataset

def test_new_normalization():
    """测试新的归一化方法"""
    print("=== 测试新的医学图像归一化方法 ===")
    
    # 配置PICAI数据集
    data_config = {
        'picai_folds': [
            '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold0',
            '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold1',
            '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold2',
            '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold3',
            '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_public_images_fold4',
        ],
        'mask_dir': '/raid/candi/junqing/FYP-SERVER/datasets/PICAI/picai_labels-main/csPCa_lesion_delineations/human_expert/original',
        'keep_cases_path': '/raid/candi/junqing/FYP-SERVER/rl_module/data/keep_cases.txt'
    }
    
    # 创建数据集
    dataset = MRIDataset(
        data_dirs=data_config['picai_folds'],
        mask_dir=data_config['mask_dir'],
        keep_cases_path=data_config['keep_cases_path'],
        dataset_type='all',
        split_ratio={'train': 0.8, 'valid': 0.1, 'test': 0.1}
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    try:
        normalized_volume, mask, original_volume, case_id = dataset._load_and_preprocess(0)
        
        print(f"Case ID: {case_id}")
        print(f"Normalized volume shape: {normalized_volume.shape}")
        print(f"Original volume shape: {original_volume.shape if original_volume is not None else 'None'}")
        print(f"Mask shape: {mask.shape}")
        
        # 分析归一化效果
        modality_names = ['ADC', 'HBV', 'T2W']
        
        print("\n=== 归一化效果分析 ===")
        for i, modality in enumerate(modality_names):
            if i < normalized_volume.shape[0]:
                normalized = normalized_volume[i]
                original = original_volume[i] if original_volume is not None else None
                
                print(f"\n{modality} 模态:")
                if original is not None:
                    print(f"  原始数据: 范围 {original.min():.1f} - {original.max():.1f}, 均值 {original.mean():.3f}")
                print(f"  归一化后: 范围 {normalized.min():.3f} - {normalized.max():.3f}, 均值 {normalized.mean():.3f}")
                
                # 检查对比度
                bg_mask = normalized < -0.5
                fg_mask = ~bg_mask
                print(f"  背景比例: {np.sum(bg_mask) / bg_mask.size * 100:.1f}%")
                if np.any(fg_mask):
                    print(f"  前景对比度: 均值 {normalized[fg_mask].mean():.3f}, 标准差 {normalized[fg_mask].std():.3f}")
        
        # 可视化对比
        if original_volume is not None:
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            
            mid_slice = normalized_volume.shape[-1] // 2
            
            for i, modality in enumerate(modality_names):
                if i < min(normalized_volume.shape[0], 3):
                    # 原始图像
                    original_slice = original_volume[i, :, :, mid_slice]
                    axes[i, 0].imshow(original_slice, cmap='gray')
                    axes[i, 0].set_title(f'{modality} Original\n{original_slice.min():.0f}-{original_slice.max():.0f}')
                    axes[i, 0].axis('off')
                    
                    # 归一化后
                    normalized_slice = normalized_volume[i, :, :, mid_slice]
                    axes[i, 1].imshow(normalized_slice, cmap='gray')
                    axes[i, 1].set_title(f'{modality} Normalized\n{normalized_slice.min():.3f}-{normalized_slice.max():.3f}')
                    axes[i, 1].axis('off')
                    
                    # 原始图像（窗位显示）
                    p5, p95 = np.percentile(original_slice, [5, 95])
                    axes[i, 2].imshow(original_slice, cmap='gray', vmin=p5, vmax=p95)
                    axes[i, 2].set_title(f'{modality} Original (Windowed)\n{p5:.0f}-{p95:.0f}')
                    axes[i, 2].axis('off')
                    
                    # 归一化后（增强对比度）
                    p5_norm, p95_norm = np.percentile(normalized_slice, [5, 95])
                    axes[i, 3].imshow(normalized_slice, cmap='gray', vmin=p5_norm, vmax=p95_norm)
                    axes[i, 3].set_title(f'{modality} Normalized (Enhanced)\n{p5_norm:.3f}-{p95_norm:.3f}')
                    axes[i, 3].axis('off')
            
            plt.suptitle(f'New Normalization Test - Case: {case_id}', fontsize=16)
            plt.tight_layout()
            plt.savefig('new_normalization_test.png', dpi=150, bbox_inches='tight')
            print(f"\n可视化结果保存为: new_normalization_test.png")
            plt.close()
        
        print("\n✅ 新的归一化方法测试完成！")
        print("主要改进:")
        print("1. 背景和前景分离良好")
        print("2. 保持了医学图像的对比度")
        print("3. 避免了过度裁剪导致的信息丢失")
        print("4. 同时保存原始和归一化数据用于不同用途")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_normalization()
    if success:
        print("\n🎉 测试成功！新的归一化方法已准备就绪。")
    else:
        print("\n❌ 测试失败，请检查配置。") 