import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from typing import Tuple, Optional, Dict, List
from scipy.ndimage import zoom
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_keep_cases(path: str) -> set:
    with open(path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

class MRIDataset:
    """Dataset class for loading MRI data and masks, now supports PICAI multi-fold structure and keep_cases filtering."""
    
    def __init__(self, 
                 data_dirs: List[str] = None, 
                 mask_dir: str = None, 
                 keep_cases_path: str = None, 
                 dataset_type: str = 'all', 
                 split_ratio: Dict = None, 
                 seed: int = 42):
        """
        Args:
            data_dirs: List of directories containing patient data (for PICAI, five fold dirs)
            mask_dir: Directory containing mask files (for PICAI)
            keep_cases_path: Path to keep_cases.txt (for PICAI)
            dataset_type: 'train', 'valid', 'test', or 'all'
            split_ratio: Dictionary containing train/valid/test split ratios
            seed: Random seed for reproducibility
        """
        self.data_dirs = data_dirs
        self.mask_dir = mask_dir
        self.keep_cases_path = keep_cases_path
        self.dataset_type = dataset_type
        self.split_ratio = split_ratio or {'train': 0.7, 'valid': 0.15, 'test': 0.15}
        np.random.seed(seed)

        self.samples = []
        self.patient_status = {}

        if self.keep_cases_path and self.data_dirs and self.mask_dir:
            # PICAI新数据集模式
            keep_cases = load_keep_cases(self.keep_cases_path)
            for fold_dir in self.data_dirs:
                for patient_id in os.listdir(fold_dir):
                    patient_path = os.path.join(fold_dir, patient_id)
                    if not os.path.isdir(patient_path):
                        continue
                    for file in os.listdir(patient_path):
                        if file.endswith('_t2w.mha'):
                            case_id = file.replace('_t2w.mha', '')
                            if case_id in keep_cases:
                                self.samples.append({
                                    'fold_dir': fold_dir,
                                    'patient_id': patient_id,
                                    'case_id': case_id
                                })
                                # 判断是否阳性
                                mask_path = os.path.join(self.mask_dir, f"{case_id}.nii.gz")
                                if os.path.exists(mask_path):
                                    mask = nib.load(mask_path).get_fdata()
                                    self.patient_status[case_id] = 'diseased' if np.any(mask > 0) else 'healthy'
                                else:
                                    self.patient_status[case_id] = 'healthy'
            # 划分数据集
            if dataset_type != 'all':
                np.random.shuffle(self.samples)
                n_total = len(self.samples)
                n_train = int(n_total * self.split_ratio['train'])
                n_valid = int(n_total * self.split_ratio['valid'])
                if dataset_type == 'train':
                    self.samples = self.samples[:n_train]
                elif dataset_type == 'valid':
                    self.samples = self.samples[n_train:n_train+n_valid]
                elif dataset_type == 'test':
                    self.samples = self.samples[n_train+n_valid:]
                else:
                    raise ValueError(f"Unknown dataset type: {dataset_type}")
            logger.info(f"[PICAI] Dataset type: {dataset_type}")
            logger.info(f"[PICAI] Found {len(self.samples)} valid cases")
            num_diseased = sum(1 for s in self.samples if self.patient_status.get(s['case_id']) == 'diseased')
            num_healthy = len(self.samples) - num_diseased
            logger.info(f"[PICAI] Diseased: {num_diseased}, Healthy: {num_healthy}")
        else:
            # 兼容原有数据集结构
            all_patient_dirs = []
            for d in os.listdir(self.data_dir):
                patient_dir = os.path.join(self.data_dir, d)
                if not os.path.isdir(patient_dir):
                    continue
                required_modalities = ['adc.nii.gz', 'dwi.nii.gz', 't2.nii.gz']
                has_all_modalities = all(
                    os.path.exists(os.path.join(patient_dir, modality))
                    for modality in required_modalities
                )
                if has_all_modalities:
                    all_patient_dirs.append(d)
                    has_cancer = os.path.exists(os.path.join(patient_dir, 'l_a1.nii.gz'))
                    self.patient_status[d] = 'diseased' if has_cancer else 'healthy'
            if dataset_type != 'all':
                np.random.shuffle(all_patient_dirs)
                n_total = len(all_patient_dirs)
                n_train = int(n_total * self.split_ratio['train'])
                n_valid = int(n_total * self.split_ratio['valid'])
                if dataset_type == 'train':
                    self.patient_dirs = all_patient_dirs[:n_train]
                elif dataset_type == 'valid':
                    self.patient_dirs = all_patient_dirs[n_train:n_train+n_valid]
                elif dataset_type == 'test':
                    self.patient_dirs = all_patient_dirs[n_train+n_valid:]
                else:
                    raise ValueError(f"Unknown dataset type: {dataset_type}")
            else:
                self.patient_dirs = all_patient_dirs
            num_diseased = sum(1 for p in self.patient_dirs if self.patient_status.get(p) == 'diseased')
            num_healthy = len(self.patient_dirs) - num_diseased
            logger.info(f"Dataset type: {dataset_type}")
            logger.info(f"Found {len(self.patient_dirs)} valid patients")
            logger.info(f"Diseased patients: {num_diseased}")
            logger.info(f"Healthy patients: {num_healthy}")

    def _medical_normalize(self, data, method='hybrid_normalize'):
        """
        医学图像专用归一化方法
        
        Args:
            data: 输入图像数据
            method: 归一化方法
        """
        # 移除背景（假设背景是0或接近0的值）
        background_threshold = np.percentile(data, 5)  # 前5%作为背景
        foreground_mask = data > background_threshold
        
        if method == 'hybrid_normalize':
            # 混合归一化（推荐）
            # 第一步：移除极值
            p2, p98 = np.percentile(data, [2, 98])
            clipped_data = np.clip(data, p2, p98)
            
            # 第二步：分别处理背景和前景
            if np.any(foreground_mask):
                # 前景区域使用百分位数归一化
                foreground_data = clipped_data[foreground_mask]
                fg_min, fg_max = np.percentile(foreground_data, [5, 95])
                
                normalized = np.copy(clipped_data).astype(np.float32)
                
                # 前景归一化到[0, 1]
                if fg_max > fg_min:
                    normalized[foreground_mask] = (
                        (foreground_data - fg_min) / (fg_max - fg_min + 1e-8)
                    )
                
                # 背景保持低值
                normalized[~foreground_mask] = 0.0
                
                # 最终映射到[-1, 1]范围，但保持对比度
                # 背景 -> [-1, -0.5], 前景 -> [-0.5, 1]
                normalized[~foreground_mask] = -1.0 + normalized[~foreground_mask] * 0.5
                normalized[foreground_mask] = -0.5 + normalized[foreground_mask] * 1.5
            else:
                # 如果没有明显前景，使用简单归一化
                normalized = (clipped_data - clipped_data.min()) / (clipped_data.max() - clipped_data.min() + 1e-8)
                normalized = 2 * normalized - 1
        else:
            # 原始方法（用于对比）
            clipped_data = np.clip(data, -5, 5)
            normalized = 2 * (clipped_data - clipped_data.min()) / (clipped_data.max() - clipped_data.min() + 1e-8) - 1
        
        return normalized.astype(np.float32)

    def _load_and_preprocess(self, idx_or_id):
        if hasattr(self, 'samples') and self.samples:
            # PICAI模式
            if isinstance(idx_or_id, int):
                sample = self.samples[idx_or_id]
            else:
                sample = next(s for s in self.samples if s['case_id'] == idx_or_id)
            case_id = sample['case_id']
            patient_path = os.path.join(sample['fold_dir'], sample['patient_id'])
            # 读取影像
            modalities = []
            original_modalities = []  # 保存原始数据用于可视化
            shapes = []
            for suffix in ['_adc.mha', '_hbv.mha', '_t2w.mha']:
                img_path = os.path.join(patient_path, f"{case_id}{suffix}")
                if img_path.endswith('.mha'):
                    img_original = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                    img_original = np.transpose(img_original, (2, 1, 0))
                else:
                    img_original = nib.load(img_path).get_fdata()
                
                # 保存原始数据
                original_modalities.append(img_original.copy())
                
                # 使用改进的医学图像归一化方法
                img_normalized = self._medical_normalize(img_original)
                shapes.append(img_normalized.shape)
                modalities.append(img_normalized)
            # 对齐shape（对归一化数据和原始数据都进行对齐）
            if len(set(str(shape) for shape in shapes)) > 1:
                min_shape = np.min(shapes, axis=0)
                
                # 对齐归一化数据
                resampled_modalities = []
                for img in modalities:
                    if img.shape != tuple(min_shape):
                        scale_factors = min_shape / np.array(img.shape)
                        resampled = zoom(img, scale_factors, order=1)
                        resampled_modalities.append(resampled)
                    else:
                        resampled_modalities.append(img)
                modalities = resampled_modalities
                
                # 对齐原始数据
                resampled_original_modalities = []
                for img_original in original_modalities:
                    if img_original.shape != tuple(min_shape):
                        scale_factors = min_shape / np.array(img_original.shape)
                        resampled_original = zoom(img_original, scale_factors, order=1)
                        resampled_original_modalities.append(resampled_original)
                    else:
                        resampled_original_modalities.append(img_original)
                original_modalities = resampled_original_modalities
            # 归一化后统计
            if idx_or_id is not None and isinstance(idx_or_id, int) and idx_or_id < 10:
                for i, mod in enumerate(modalities):
                    print(f"[DEBUG] Case {case_id} modality {i}: min={mod.min():.3f}, max={mod.max():.3f}, mean={mod.mean():.3f}, std={mod.std():.3f}")
            volume = np.stack(modalities, axis=0)
            # 读取mask
            mask_path = os.path.join(self.mask_dir, f"{case_id}.nii.gz")
            if os.path.exists(mask_path):
                mask = nib.load(mask_path).get_fdata()
                if mask.shape != volume.shape[1:]:
                    scale_factors = np.array(volume.shape[1:]) / np.array(mask.shape)
                    mask = zoom(mask, scale_factors, order=0)
                mask = (mask > 0.5).astype(np.float32)
            else:
                mask = np.zeros(volume.shape[1:], dtype=np.float32)
            if idx_or_id is not None and isinstance(idx_or_id, int) and idx_or_id < 10:
                print(f"[DEBUG] Case {case_id} mask: min={mask.min():.3f}, max={mask.max():.3f}, mean={mask.mean():.3f}, std={mask.std():.3f}")
            
            # 将原始数据作为额外信息返回
            original_volume = np.stack(original_modalities, axis=0) if original_modalities else None
            return volume, mask, original_volume, case_id
        else:
            # 兼容原有数据集
            volume, mask = self._load_and_preprocess_orig(idx_or_id)
            return volume, mask, None, idx_or_id  # 保持返回格式一致

    def _load_and_preprocess_orig(self, patient_id: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        patient_dir = os.path.join(self.data_dir, patient_id)
        modalities = []
        shapes = []
        for modality in ['adc.nii.gz', 'dwi.nii.gz', 't2.nii.gz']:
            img_path = os.path.join(patient_dir, modality)
            img = nib.load(img_path).get_fdata()
            shapes.append(img.shape)
            modalities.append(img)
        if len(set(str(shape) for shape in shapes)) > 1:
            min_shape = np.min(shapes, axis=0)
            resampled_modalities = []
            for img in modalities:
                if img.shape != tuple(min_shape):
                    scale_factors = min_shape / np.array(img.shape)
                    resampled = zoom(img, scale_factors, order=1)
                    resampled_modalities.append(resampled)
                else:
                    resampled_modalities.append(img)
            modalities = resampled_modalities
        normalized_modalities = []
        for mod in modalities:
            p1, p99 = np.percentile(mod, (1, 99))
            mod_norm = np.clip(mod, p1, p99)
            mod_norm = (mod_norm - mod_norm.mean()) / (mod_norm.std() + 1e-8)
            normalized_modalities.append(mod_norm)
        volume = np.stack(normalized_modalities, axis=0)
        mask = None
        if self.patient_status[patient_id] == 'diseased':
            mask_path = os.path.join(patient_dir, 'l_a1.nii.gz')
            mask = nib.load(mask_path).get_fdata()
            if mask.shape != volume.shape[1:]:
                scale_factors = np.array(volume.shape[1:]) / np.array(mask.shape)
                mask = zoom(mask, scale_factors, order=0)
            mask = (mask > 0.5).astype(np.float32)
        else:
            mask = np.zeros(volume.shape[1:], dtype=np.float32)
        return volume, mask

    def __len__(self):
        if hasattr(self, 'samples') and self.samples:
            return len(self.samples)
        else:
            return len(self.patient_dirs)

    def get_random_patient(self):
        idx = np.random.randint(len(self))
        volume, mask, original_volume, case_id = self._load_and_preprocess(idx)
        return volume, mask, idx, original_volume, case_id

    def get_specific_patient(self, case_id: str):
        volume, mask, original_volume, _ = self._load_and_preprocess(case_id)
        # 尝试找到对应的索引
        idx = -1
        if hasattr(self, 'samples') and self.samples:
            for i, sample in enumerate(self.samples):
                if sample['case_id'] == case_id:
                    idx = i
                    break
        elif hasattr(self, 'patient_dirs'):
            try:
                idx = self.patient_dirs.index(case_id)
            except ValueError:
                idx = -1
        return volume, mask, idx, original_volume, case_id

    def get_patient_status(self, case_id: str) -> str:
        return self.patient_status.get(case_id, 'unknown') 