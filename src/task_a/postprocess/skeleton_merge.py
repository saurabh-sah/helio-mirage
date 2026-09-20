import cv2
import numpy as np
from skimage.morphology import skeletonize

def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Skeletonizes a binary mask.
    Args:
        mask: Binary numpy array (0 and 255)
    Returns:
        Binary numpy array skeleton (0 and 1)
    """
    # skeletonize expects boolean or 0/1 array
    bool_mask = mask > 127
    skeleton = skeletonize(bool_mask)
    return skeleton.astype(np.uint8)

def graph_merge_fragments(mask: np.ndarray, distance_threshold: int = 15) -> np.ndarray:
    """
    Merges disconnected filament fragments into single coherent objects based on proximity.
    This acts as the post-hoc instance coherence step described in Phase 4.
    
    Args:
        mask: Binary numpy array (0 and 255) containing filament predictions
        distance_threshold: Maximum distance in pixels to connect nearby fragments.
                            This hyperparameter should be tied to GONG pixel scale 
                            and physical barb continuity statistics.
                            
    Returns:
        Merged binary mask
    """
    # 1. We can simulate a distance-based merge by dilating the mask by the distance threshold,
    # finding connected components, and then retaining the original mask pixels but giving them
    # the labels of the dilated components.
    
    kernel_size = distance_threshold
    if kernel_size % 2 == 0:
        kernel_size += 1 # Needs to be odd for symmetric kernel
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Dilate to connect nearby fragments
    dilated = cv2.dilate(mask, kernel, iterations=1)
    
    # Find connected components on the dilated mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    # Create an instance mask where original pixels get the ID of their dilated merged blob
    instance_mask = np.zeros_like(mask, dtype=np.int32)
    
    # Only keep the labels for pixels that were in the original mask
    original_pixels_idx = mask > 127
    instance_mask[original_pixels_idx] = labels[original_pixels_idx]
    
    # We can also return a binary mask of the successfully merged components
    # (Here we just return a binarized version of the cleaned mask for standard IoU)
    # But usually, instance segmentation requires the instance IDs.
    
    # Let's clean up tiny isolated dots that didn't merge into anything substantial
    min_area = 50
    final_binary_mask = np.zeros_like(mask)
    
    # We count area of the original pixels within each instance, not the dilated area
    for i in range(1, num_labels):
        instance_pixels = (instance_mask == i)
        area = np.count_nonzero(instance_pixels)
        if area >= min_area:
            final_binary_mask[instance_pixels] = 255
            
    return final_binary_mask
