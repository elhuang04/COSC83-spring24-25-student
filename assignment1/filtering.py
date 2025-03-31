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
    # TODO: Implement the 2D convolution operation
    # 1. Check if kernel dimensions are odd
    # 2. Handle both 2D (grayscale) and 3D (color) images
    # 3. Calculate appropriate padding
    # 4. Create output image
    # 5. Apply convolution for each channel
    # 6. Return the result in the same shape as input
    
    pass  # Replace with your implementation

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
    # TODO: Implement the mean filter
    # 1. Create a mean filter kernel of size kernel_size × kernel_size
    # 2. Apply the convolution using the convolve2d function
    
    pass  # Replace with your implementation

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
    # TODO: Implement the Gaussian kernel generation
    # 1. Check if kernel size is odd
    # 2. Generate grid coordinates centered at 0
    # 3. Compute the Gaussian kernel based on the formula
    # 4. Normalize the kernel so it sums to 1
    
    pass  # Replace with your implementation

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
    # TODO: Implement the Gaussian filter
    # 1. Generate a Gaussian kernel using the gaussian_kernel function
    # 2. Apply convolution using the convolve2d function
    
    pass  # Replace with your implementation

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
    # TODO: Implement the Laplacian filter
    # 1. Define the appropriate Laplacian kernel based on kernel_type
    # 2. Apply convolution using the convolve2d function
    
    pass  # Replace with your implementation

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
    # TODO: Implement the Sobel filter
    # 1. Define Sobel kernels in x and y directions based on kernel_size
    # 2. Apply convolution based on the specified direction
    # 3. For 'both' direction, compute gradient magnitude and direction
    # 4. Return appropriate output based on direction parameter
    
    pass  # Replace with your implementation

# These helper functions are provided for you

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to range [0, 255] and convert to uint8.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
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