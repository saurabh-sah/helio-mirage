import cv2
import numpy as np

def apply_clahe(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    only within the provided mask (e.g., the solar disk).
    
    Args:
        image: Grayscale image array
        mask: Binary mask where 1/True is the region of interest
        
    Returns:
        Contrast enhanced image array
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    if mask is None:
        return clahe.apply(image)
        
    # Apply CLAHE to the whole image
    enhanced = clahe.apply(image)
    
    # Only keep the enhanced pixels within the mask, zero out the rest
    result = np.zeros_like(image)
    result[mask] = enhanced[mask]
    
    return result
