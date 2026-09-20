import numpy as np
import cv2

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Computes standard Intersection over Union for boolean masks."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Computes pixel-level precision and recall."""
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

def compute_miou(pred: np.ndarray, gt: np.ndarray, max_dilation: int = 5) -> float:
    """
    Multi-scale IoU (MIoU) implementation.
    The exact formula was not provided in the problem statement.
    This is an operationalization: we compute IoU at multiple morphological dilations
    of both GT and Prediction, averaging across scales. This rewards the network for 
    capturing thin structures even if they are slightly shifted by a pixel or two.
    """
    iou_scores = []
    
    # Scale 0 is standard IoU
    iou_scores.append(compute_iou(pred, gt))
    
    pred_uint = pred.astype(np.uint8) * 255
    gt_uint = gt.astype(np.uint8) * 255
    
    for i in range(1, max_dilation + 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*i + 1, 2*i + 1))
        pred_dilated = cv2.dilate(pred_uint, kernel) > 127
        gt_dilated = cv2.dilate(gt_uint, kernel) > 127
        
        iou_scores.append(compute_iou(pred_dilated, gt_dilated))
        
    # Average across all scales
    return sum(iou_scores) / len(iou_scores)

def compute_instance_metrics(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """
    Computes Hit Rate, Miss Rate, and AP at a specific IoU threshold (AP@IoU).
    Works on instance level.
    """
    # Find instances in GT and Pred
    num_pred, labels_pred = cv2.connectedComponents(pred.astype(np.uint8) * 255, connectivity=8)
    num_gt, labels_gt = cv2.connectedComponents(gt.astype(np.uint8) * 255, connectivity=8)
    
    if num_gt <= 1: # Only background
        return {"hit_rate": 0.0, "miss_rate": 0.0, "ap": 0.0 if num_pred > 1 else 1.0}
        
    hits = 0
    gt_matched = set()
    
    for i in range(1, num_pred):
        pred_mask = (labels_pred == i)
        best_iou = 0
        best_gt_idx = -1
        
        # Check against all GT instances
        # (In practice, optimize by finding overlapping bounding boxes first)
        for j in range(1, num_gt):
            if j in gt_matched:
                continue
            gt_mask = (labels_gt == j)
            
            iou = compute_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
                
        if best_iou >= iou_threshold:
            hits += 1
            gt_matched.add(best_gt_idx)
            
    total_gt_instances = num_gt - 1
    total_pred_instances = num_pred - 1
    
    hit_rate = hits / total_gt_instances if total_gt_instances > 0 else 0.0
    miss_rate = 1.0 - hit_rate
    
    # AP is simplified here to precision at the threshold
    ap = hits / total_pred_instances if total_pred_instances > 0 else 0.0
    
    return {
        "hit_rate": hit_rate,
        "miss_rate": miss_rate,
        f"ap@{iou_threshold}": ap
    }
