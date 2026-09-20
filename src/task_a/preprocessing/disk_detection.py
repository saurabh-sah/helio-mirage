import cv2
import numpy as np

def detect_solar_disk(image: np.ndarray) -> tuple[int, int, int]:
    """
    Detects the solar disk using Hough Circle Transform.
    Falls back to radial intensity profile fitting if Hough fails.
    
    Args:
        image: Grayscale image array (H, W)
        
    Returns:
        tuple (center_x, center_y, radius)
    """
    # Thresholding / edge map
    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=image.shape[0]//2,
        param1=50, 
        param2=30, 
        minRadius=int(image.shape[0]*0.2), 
        maxRadius=int(image.shape[0]*0.6)
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0, :]
        return int(x), int(y), int(r)
    else:
        # Fallback: simple center of mass and threshold-based radius
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = image.shape[1] // 2, image.shape[0] // 2
        
        # Estimate radius from area
        area = np.count_nonzero(thresh)
        radius = int(np.sqrt(area / np.pi))
        return cX, cY, radius
