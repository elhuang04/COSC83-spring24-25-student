import numpy as np
from filtering import gaussian_filter, sobel_filter

import cv2

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

    gaussian = gaussian_filter(image, 5, sigma)
    magnitude, angle = sobel_filter(gaussian, direction='both')
    angle = np.rad2deg(angle) % 180

    max_val = magnitude.max()
    high = max_val * high_threshold
    low = max_val * low_threshold

    output = np.zeros_like(magnitude, dtype=np.uint8)
    strong, weak = 255, 75

    increment = 180/8
    increments = [i*increment for i in range (9)]

    print(increments)

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            theta = angle[i, j]
            mag = magnitude[i, j]

            if (increments[0] <= theta < increments[1]) or (increments[-2] <= theta < increments[-1]):
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
            elif increments[1] <= theta < increments[3]:
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
            elif increments[3] <= theta < increments[5]:
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
            elif increments[5] <= theta < increments[7]:
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            # non maximum supression
            if mag >= n1 and mag >= n2:
                if mag >= high:
                    output[i, j] = strong
                elif mag >= low:
                    output[i, j] = weak

    # find neighboring connected edges
    for i in range(1, output.shape[0] - 1):
        for j in range(1, output.shape[1] - 1):
            if output[i, j] == weak:
                neighborhood = output[i-1:i+2, j-1:j+2]
                if np.any(neighborhood == strong):
                    output[i, j] = strong
                else:
                    output[i, j] = 0

    return output


# image = cv2.imread("assignment1/example_images/50px.jpg")
# # Convert to RGB for display
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # Convert to grayscale for edge detection
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# canny_edge_detector(gray)
