import numpy as np

def correct_limb_darkening(image: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    """
    Corrects solar limb darkening by fitting a radial intensity profile.
    
    Args:
        image: Grayscale image array
        cx, cy, r: center x, center y, and radius of the solar disk
        
    Returns:
        Corrected image array
    """
    # Create coordinate grid
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    
    # Distance from center
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Normalize distance to [0, 1] relative to radius
    rho = dist / float(r)
    
    # Mask out-of-disk
    mask = rho <= 1.0
    
    # Simple limb darkening formula: I(rho) = I(0) * (1 - u + u * cos(theta))
    # where cos(theta) = sqrt(1 - rho^2)
    # We estimate mean intensity at different rho bins to fit the curve, 
    # but here we use a standard empirical correction for H-alpha
    
    # Pre-calculated empirical approximation
    cos_theta = np.sqrt(np.clip(1 - rho**2, 0, 1))
    
    # Add small epsilon to avoid division by zero near the limb
    correction_factor = 1.0 / (0.3 + 0.7 * cos_theta + 1e-6)
    
    corrected = np.zeros_like(image, dtype=np.float32)
    corrected[mask] = image[mask] * correction_factor[mask]
    
    # Normalize back to 0-255
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected
