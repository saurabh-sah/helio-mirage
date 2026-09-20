import cv2
import numpy as np

def classical_segmentation_pipeline(image: np.ndarray, disk_mask: np.ndarray) -> np.ndarray:
    """
    Classical image processing baseline for solar filament segmentation.
    This serves as the absolute minimum floor to beat, as requested in Phase 1.
    
    Args:
        image: Preprocessed H-alpha image (limb-darkening corrected and CLAHE applied)
        disk_mask: Binary mask of the solar disk
        
    Returns:
        Binary segmentation mask of filaments (255 for filaments, 0 for background)
    """
    # 1. Apply slight Gaussian blur to reduce high-frequency noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 2. Adaptive thresholding. 
    # Filaments are dark structures, so we threshold to find regions darker than neighborhood.
    # The block size should roughly correspond to the typical filament width + margin.
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 45, 10
    )
    
    # 3. Mask out non-disk regions (space and off-limb noise)
    if disk_mask is not None:
        thresh = cv2.bitwise_and(thresh, thresh, mask=disk_mask)
        
    # 4. Morphological operations to clean up noise and connect fragments
    # Opening removes small bright noise (which in our inverted image are small dark spots in reality)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
    
    # Closing connects nearby fragments, critical for maintaining filament coherence
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)
    
    # 5. Filter by area (connected components) to remove small spurious islands
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    final_mask = np.zeros_like(closed)
    
    # The threshold for minimum area should ideally be tuned based on image resolution.
    # E.g., for a 2048x2048 image, 100 pixels is tiny.
    min_area = 50 
    
    for i in range(1, num_labels): # Skip 0 which is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            final_mask[labels == i] = 255
            
    return final_mask
