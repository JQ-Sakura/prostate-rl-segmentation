import os
import pandas as pd
import nibabel as nib
import numpy as np

# 路径配置
marksheet_path = 'E:/Study/UCL/FYP-SERVER/datasets/PICAI/picai_labels-main/clinical_information/marksheet.csv'
mask_dir = 'E:/Study/UCL/FYP-SERVER/datasets/PICAI/picai_labels-main/csPCa_lesion_delineations/human_expert/original'
image_folds = [
    'E:/Study/UCL/FYP-SERVER/datasets/PICAI/picai_public_images_fold0',
    'E:/Study/UCL/FYP-SERVER/datasets/PICAI/picai_public_images_fold1',
    'E:/Study/UCL/FYP-SERVER/datasets/PICAI/picai_public_images_fold2',
    'E:/Study/UCL/FYP-SERVER/datasets/PICAI/picai_public_images_fold3',
    'E:/Study/UCL/FYP-SERVER/datasets/PICAI/picai_public_images_fold4'
]

# 读取临床信息
df = pd.read_csv(marksheet_path)
keep_cases = []

for idx, row in df.iterrows():
    patient_id = str(row['patient_id'])
    study_id = str(row['study_id'])
    case_id = f"{patient_id}_{study_id}"
    csPCa = str(row['case_csPCa']).strip().upper()
    mask_path = os.path.join(mask_dir, f"{case_id}.nii.gz")
    
    if csPCa == 'NO':
        # 阴性病例全部保留
        keep_cases.append(case_id)
    elif csPCa == 'YES':
        # 阳性病例需检查mask内容
        if os.path.exists(mask_path):
            mask = nib.load(mask_path).get_fdata()
            if np.any(mask > 0):  # 有注释
                keep_cases.append(case_id)
        else:
            print(f"Warning: mask not found for {case_id}")

# 可选：保存keep_cases到文件
with open('keep_cases.txt', 'w') as f:
    for case in keep_cases:
        f.write(case + '\n')

print(f"Total cases to keep: {len(keep_cases)}")
