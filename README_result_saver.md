# 训练结果保存功能使用说明

## 功能概述

训练过程中自动保存最好的K个分割结果，包括：
- 预测分割掩码
- 真实分割掩码  
- 三模态的病人扫描图（ADC、HBV、T2W）
- 详细的元数据信息

## 配置说明

在 `rl_module/configs/train_config.yaml` 中可以配置：

```yaml
training:
  # 结果保存配置
  results_save_dir: '/raid/candi/junqing/FYP-SERVER/training_results'  # 保存目录
  save_top_k_results: 10  # 保存最好的K个结果
  save_threshold_dice: 0.1  # 只保存Dice分数大于此阈值的结果
```

## 保存的目录结构

```
training_results/
├── predictions/           # 预测掩码文件
│   ├── ep100_st5_case001_dice0.7234_pred.nii.gz
│   └── ...
├── ground_truth/         # 真实掩码文件
│   ├── ep100_st5_case001_dice0.7234_gt.nii.gz
│   └── ...
├── volumes/              # 三模态扫描图
│   ├── ep100_st5_case001_dice0.7234_adc.nii.gz
│   ├── ep100_st5_case001_dice0.7234_hbv.nii.gz
│   ├── ep100_st5_case001_dice0.7234_t2w.nii.gz
│   └── ...
├── metadata/             # 详细元数据
│   ├── ep100_st5_case001_dice0.7234_metadata.json
│   └── ...
└── top_results_metadata.json  # 总体元数据文件
```

## 文件命名规则

文件名格式：`ep{episode}_st{step}_{case_id}_dice{dice_score:.4f}_{type}.nii.gz`

- `episode`: 训练轮次
- `step`: 步骤
- `case_id`: 病例ID
- `dice_score`: Dice分数（4位小数）
- `type`: 文件类型（pred/gt/adc/hbv/t2w）

## 查看保存的结果

### 1. 使用脚本查看结果摘要

```bash
# 查看结果摘要
python view_saved_results.py /raid/candi/junqing/FYP-SERVER/training_results --summary

# 可视化排名第1的结果
python view_saved_results.py /raid/candi/junqing/FYP-SERVER/training_results --visualize 1

# 可视化排名第1的结果的第50个切片
python view_saved_results.py /raid/candi/junqing/FYP-SERVER/training_results --visualize 1 --slice 50

# 可视化并保存图像
python view_saved_results.py /raid/candi/junqing/FYP-SERVER/training_results --visualize 1 --save result_1.png
```

### 2. 直接加载数据进行分析

```python
from rl_module.utils.result_saver import TopResultsSaver
import nibabel as nib

# 创建结果查看器
saver = TopResultsSaver('/raid/candi/junqing/FYP-SERVER/training_results')

# 查看摘要
saver.print_summary()

# 获取最好的结果
top_results = saver.get_top_results()
best_result = top_results[0]

print(f"最好结果: Episode {best_result.episode}, Dice: {best_result.dice_score:.4f}")
```

### 3. 手动加载特定文件

```python
import nibabel as nib
import json

# 加载预测掩码
pred_mask = nib.load('training_results/predictions/ep100_st5_case001_dice0.7234_pred.nii.gz').get_fdata()

# 加载真实掩码
gt_mask = nib.load('training_results/ground_truth/ep100_st5_case001_dice0.7234_gt.nii.gz').get_fdata()

# 加载T2W图像
t2w_volume = nib.load('training_results/volumes/ep100_st5_case001_dice0.7234_t2w.nii.gz').get_fdata()

# 加载元数据
with open('training_results/metadata/ep100_st5_case001_dice0.7234_metadata.json', 'r') as f:
    metadata = json.load(f)
    
print(f"Dice分数: {metadata['dice_score']}")
print(f"IoU分数: {metadata['iou_score']}")
print(f"病例ID: {metadata['case_id']}")
```

## 训练中的输出信息

训练过程中会显示类似以下的信息：

```
📁 Results will be saved to: /raid/candi/junqing/FYP-SERVER/training_results
🏆 Saving top 10 segmentation results
✅ Saved result: Episode 100, Step 5, Dice: 0.7234, IoU: 0.6543
✅ Saved result: Episode 150, Step 3, Dice: 0.7456, IoU: 0.6789
🗑️  Removed files for result: Episode 50, Dice: 0.5123
```

每100个episode会打印一次结果摘要：

```
📊 Top 10 Segmentation Results:
================================================================================
Rank  Episode  Step   Dice     IoU      Case ID         Timestamp
--------------------------------------------------------------------------------
1     150      3      0.7456   0.6789   case001         2024-01-15 10:30:45
2     100      5      0.7234   0.6543   case002         2024-01-15 10:25:12
...
================================================================================
Best Dice Score: 0.7456
Average Dice Score: 0.6892
Results saved in: /raid/candi/junqing/FYP-SERVER/training_results
```

## 注意事项

1. **存储空间**: 每个结果包含5个文件（预测、真实、3个模态），注意磁盘空间使用
2. **性能影响**: 保存操作在后台进行，对训练速度影响很小
3. **阈值设置**: 通过 `save_threshold_dice` 避免保存质量过低的结果
4. **自动清理**: 当保存的结果超过 `save_top_k_results` 时，会自动删除最差的结果

## 可能的用途

1. **模型分析**: 分析哪些类型的病例模型表现更好
2. **错误诊断**: 查看模型在哪些情况下容易出错
3. **结果展示**: 制作论文图表和演示材料
4. **进一步处理**: 后处理算法的开发和测试
5. **数据质量**: 评估数据集中不同病例的质量 