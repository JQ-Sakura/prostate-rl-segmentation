import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix

def calculate_dice(pred_mask, true_mask):
    """Calculate the Dice coefficient"""
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)
    if union == 0:
        return 0
    return 2.0 * intersection / union

def calculate_iou(pred_mask, true_mask):
    """Calculate the IoU (Intersection over Union)"""
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask) - intersection
    if union == 0:
        return 0
    return intersection / union

def calculate_hausdorff_distance(pred_mask, true_mask):
    """Calculate the Hausdorff distance"""
    pred_points = np.array(np.where(pred_mask)).T
    true_points = np.array(np.where(true_mask)).T
    
    if len(pred_points) == 0 or len(true_points) == 0:
        return float('inf')
        
    return max(directed_hausdorff(pred_points, true_points)[0],
              directed_hausdorff(true_points, pred_points)[0])

def calculate_surface_distance(pred_mask, true_mask):
    """Calculate the average surface distance"""
    from scipy.ndimage import distance_transform_edt
    
    pred_surface = pred_mask - np.logical_erosion(pred_mask)
    true_surface = true_mask - np.logical_erosion(true_mask)
    
    pred_distance = distance_transform_edt(~pred_surface)
    true_distance = distance_transform_edt(~true_surface)
    
    pred_to_true = pred_surface * true_distance
    true_to_pred = true_surface * pred_distance
    
    if np.sum(pred_surface) == 0 or np.sum(true_surface) == 0:
        return float('inf')
        
    return (np.sum(pred_to_true) + np.sum(true_to_pred)) / \
           (np.sum(pred_surface) + np.sum(true_surface))

def calculate_volume_metrics(pred_mask, true_mask):
    """Calculate the volume related metrics"""
    pred_volume = np.sum(pred_mask)
    true_volume = np.sum(true_mask)
    
    volume_difference = abs(pred_volume - true_volume)
    volume_ratio = pred_volume / true_volume if true_volume > 0 else float('inf')
    
    return {
        "volume_difference": volume_difference,
        "volume_ratio": volume_ratio
    }

def calculate_confusion_metrics(pred_mask, true_mask):
    """Calculate the confusion matrix related metrics"""
    tn, fp, fn, tp = confusion_matrix(true_mask.flatten(), 
                                    pred_mask.flatten()).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision
    }

def calculate_ssim_3d(pred_mask, true_mask):
    """Calculate the 3D SSIM"""
    ssim_scores = []
    for z in range(pred_mask.shape[0]):
        if np.any(true_mask[z]) or np.any(pred_mask[z]):
            score = ssim(pred_mask[z], true_mask[z], data_range=1)
            ssim_scores.append(score)
    
    return np.mean(ssim_scores) if ssim_scores else 0

def calculate_metrics(pred_mask, true_mask):
    """Calculate all evaluation metrics"""
    metrics = {
        "dice": calculate_dice(pred_mask, true_mask),
        "iou": calculate_iou(pred_mask, true_mask),
        "hausdorff": calculate_hausdorff_distance(pred_mask, true_mask),
        "surface_distance": calculate_surface_distance(pred_mask, true_mask),
        "ssim": calculate_ssim_3d(pred_mask, true_mask)
    }
    
    # Add volume related metrics
    volume_metrics = calculate_volume_metrics(pred_mask, true_mask)
    metrics.update(volume_metrics)
    
    # Add confusion matrix related metrics
    confusion_metrics = calculate_confusion_metrics(pred_mask, true_mask)
    metrics.update(confusion_metrics)
    
    return metrics 