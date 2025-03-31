import numpy as np
from filtering import gaussian_filter, sobel_filter

#10%bonus
def canny_edge_detector(image: np.ndarray, low_threshold: float = 0.05, high_threshold: float = 0.15, 
                      sigma: float = 1.0) -> np.ndarray:
    """
    Implement the Canny edge detection algorithm from scratch.
    
    Args:
        image: Input grayscale image
        low_threshold: Low threshold for hysteresis (as a fraction of the maximum gradient magnitude)
        high_threshold: High threshold for hysteresis (as a fraction of the maximum gradient magnitude)
        sigma: Standard deviation for Gaussian filter
        
    Returns:
        Binary edge map
    """
    # TODO: Implement the Canny edge detection algorithm
    pass  # Replace with your implementation