# Elizabeth Huang
# Last Modified: April 21, 2025

import numpy as np
from typing import Tuple, Union

# 10%
def convolve2d(image: np.ndarray, kernel: np.ndarray, padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply 2D convolution operation on an image with a given kernel.
    
    Args:
        image: Input image (2D or 3D numpy array)
        kernel: Convolution kernel (2D numpy array)
        padding_mode: How to handle borders ('constant', 'reflect', 'replicate', etc.)
        
    Returns:
        Convolved image (same size as input)
    """
    # 1. Check if kernel dimensions are odd
    # 2. Handle both 2D (grayscale) and 3D (color) images
    # 3. Calculate appropriate padding
    # 4. Create output image
    # 5. Apply convolution for each channel
    # 6. Return the result in the same shape as input
    n, m = kernel.shape  # Kernel dimensions (should be square)

    # ERROR HANDLING
    if n != m:
        raise ValueError("kernel dimensions not square")
    elif n % 2 == 0:
        raise ValueError("kernel dimensions should be odd")

    if len(image.shape) == 2:
        return convolve_single_channel(image, kernel, padding_mode)
    elif len(image.shape) == 3:
        num_channels = image.shape[2]
        
        result_channels = []
        for channel in range(num_channels):
            convolved_channel = convolve_single_channel(image[..., channel], kernel, padding_mode)
            result_channels.append(convolved_channel)
        result = np.stack(result_channels, axis=-1)
        return result

#5%
def mean_filter(image: np.ndarray, kernel_size: int = 3, padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply mean filtering to an image.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (e.g., 3 for 3x3, 5 for 5x5)
        padding_mode: How to handle borders
        
    Returns:
        Filtered image
    """
    # 1. Create a mean filter kernel of size kernel_size × kernel_size
    # 2. Apply the convolution using the convolve2d function
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    res = convolve2d(image, kernel, padding_mode)
    # print("mean summary", res.shape, res[0])
    return res

#5%
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a Gaussian kernel.
    
    Args:
        size: Kernel size (must be odd)
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Gaussian kernel (normalized)
    """
    # 1. Check if kernel size is odd
    # 2. Generate grid coordinates centered at 0
    # 3. Compute the Gaussian kernel based on the formula
    # 4. Normalize the kernel so it sums to 1
    assert size % 2 == 1, "kernel dimensions should be odd"

    center = size // 2 
    kernel = np.zeros((size, size))

    for i in range(center + 1):
        for j in range(center + 1):  
            dist_sq = (i-center)**2 + (j-center)**2
            val = (1/(2*np.pi*sigma**2))*np.exp(-dist_sq/(2*sigma**2))  #gaussian formula

            kernel[i, j] = val  
            kernel[i, size-1-j] = val  
            kernel[size-1-i, size-1-j] = val  
            kernel[size-1-i, j] = val  

    kernel /= kernel.sum()

    return kernel



#5%
def gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0, 
                   padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply Gaussian filtering to an image.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (must be odd)
        sigma: Standard deviation of the Gaussian
        padding_mode: How to handle borders
        
    Returns:
        Filtered image
    """
    # 1. Generate a Gaussian kernel using the gaussian_kernel function
    # 2. Apply convolution using the convolve2d function
    
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve2d(image, kernel, padding_mode)

#5%
def laplacian_filter(image: np.ndarray, kernel_type: str = 'standard', 
                    padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply Laplacian filtering for edge detection.
    
    Args:
        image: Input image
        kernel_type: Type of Laplacian kernel ('standard', 'diagonal')
        padding_mode: How to handle borders
        
    Returns:
        Filtered image
    """
    # 1. Define the appropriate Laplacian kernel based on kernel_type
    # 2. Apply convolution using the convolve2d function
    image = image.astype(np.float32)

    if kernel_type == "standard":
        kernel = np.array([
                            [ 0, 1,  0],
                            [1,  -4, 1],
                            [ 0, 1,  0]
                        ])
    elif kernel_type == "diagonal":
        kernel = np.array([
                            [ 1,  1, 1],
                            [ 1,  -8,  1],
                            [1,  1,  1]
                        ])
    res = convolve2d(image, kernel, padding_mode)
    # print("laplacian info:", res.shape, res[0])
    return res

#10%
def sobel_filter(image: np.ndarray, direction: str = 'both', kernel_size: int = 3, 
                padding_mode: str = 'constant') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply Sobel filtering for edge detection.
    
    Args:
        image: Input image
        direction: Direction of the filter ('x', 'y', or 'both')
        kernel_size: Size of the kernel (3, 5, etc.)
        padding_mode: How to handle borders
        
    Returns:
        If direction is 'both', returns (gradient_magnitude, gradient_direction)
        Otherwise, returns the filtered image
    """
    # 1. Define Sobel kernels in x and y directions based on kernel_size
    # 2. Apply convolution based on the specified direction
    # 3. For 'both' direction, compute gradient magnitude and direction
    # 4. Return appropriate output based on direction parameter
    image = image.copy().astype(np.float32)

    # OLD IMPLEMENTATION (computing first and second derivatives)
    # center = kernel_size // 2
    # sobel_x = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    # sobel_y = np.zeros((kernel_size, kernel_size), dtype=np.float64)

    # gaussian = gaussian_kernel(kernel_size, sigma)

    # for i in range(kernel_size):
    #     for j in range(kernel_size):
    #         dx = j - center
    #         dy = i - center

    #         sobel_x[i, j] = dx * gaussian[i, j]
    #         sobel_y[i, j] = dy * gaussian[i, j]

    # sobel_x /= np.sum(np.abs(sobel_x))
    # sobel_y /= np.sum(np.abs(sobel_y))

    if kernel_size == 3:
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])

    elif kernel_size == 5:
        sobel_x = np.array([[ 1,  2,  0, -2, -1],
                            [ 4,  8,  0, -8, -4],
                            [ 6, 12,  0, -12, -6],
                            [ 4,  8,  0, -8, -4],
                            [ 1,  2,  0, -2, -1]])

        sobel_y = np.array([[-1, -4, -6, -4, -1],
                            [-2, -8, -12, -8, -2],
                            [ 0,  0,  0,  0,  0],
                            [ 2,  8,  12,  8,  2],
                            [ 1,  4,  6,  4,  1]])

    elif kernel_size == 7:
        sobel_x = np.array([
        [-3, -2, -1,  0,  1,  2,  3],
        [-18, -12, -6,  0,  6, 12, 18],
        [-45, -30, -15,  0, 15, 30, 45],
        [-60, -40, -20,  0, 20, 40, 60],
        [-45, -30, -15,  0, 15, 30, 45],
        [-18, -12, -6,  0,  6, 12, 18],
        [ -3, -2, -1,  0,  1,  2,  3]
        ])

        sobel_y = sobel_x.T

    if direction == 'x':
        grad_x = convolve2d(image, sobel_x, padding_mode)
        return grad_x

    elif direction == 'y':
        grad_y = convolve2d(image, sobel_y, padding_mode)
        return grad_y

    elif direction == 'both':
        grad_x = convolve2d(image, sobel_x, padding_mode)
        grad_y = convolve2d(image, sobel_y, padding_mode)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        return magnitude, angle


# These helper functions are provided for you

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to range [0, 255] and convert to uint8.
    """
    min_val = np.min(image)
    # print(min_val)
    max_val = np.max(image)
    # print(max_val)
    
    # Check to avoid division by zero
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)
    
    # Normalize to [0, 255]
    normalized = 255 * (image - min_val) / (max_val - min_val)
    return normalized.astype(np.uint8)


def add_noise(image: np.ndarray, noise_type: str = 'gaussian', var: float = 0.01) -> np.ndarray:
    """
    Add noise to an image.
    
    Args:
        image: Input image
        noise_type: Type of noise ('gaussian' or 'salt_pepper')
        var: Variance (for Gaussian) or density (for salt and pepper)
        
    Returns:
        Noisy image
    """
    image_copy = image.copy().astype(np.float32)
    
    if noise_type == 'gaussian':
        # Add Gaussian noise
        noise = np.random.normal(0, var**0.5, image.shape)
        noisy = image_copy + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    elif noise_type == 'salt_pepper':
        # Add salt and pepper noise
        salt_mask = np.random.random(image.shape) < var/2
        pepper_mask = np.random.random(image.shape) < var/2
        
        noisy = image_copy.copy()
        noisy[salt_mask] = 255
        noisy[pepper_mask] = 0
        return noisy.astype(np.uint8)
    
    else:
        raise ValueError("Unknown noise type. Use 'gaussian' or 'salt_pepper'")
    
# ----------------------- HELPER FUNCTIONS ---------------------------
def convolve_single_channel(image, kernel, padding_mode):
    n, m = kernel.shape
    rows, cols = image.shape

    half_n = n // 2
    half_m = m // 2
    
    padded_img = np.pad(image,
                        ((half_n, half_n), 
                         (half_m, half_m)), 
                        mode=padding_mode)
    
    output = np.zeros_like(image)
    for row in range(half_n, rows + half_n):
        for col in range(half_m, cols + half_m):
            selection = padded_img[row-half_n:row+half_n+1, 
                                   col-half_m:col+half_m+1]
            output[row-half_n, col-half_m] = np.sum(selection * kernel)
            
    return output