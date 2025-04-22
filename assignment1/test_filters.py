# added additional tests to compare parameter variations
# Last Modified: April 21, 2025

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
from typing import List, Tuple

# Import the filtering functions
from filtering import (
    mean_filter, gaussian_filter, laplacian_filter, sobel_filter, 
    normalize_image, add_noise
)
from canny import canny_edge_detector


def test_basic_filters(image_path: str, save_path: str = None):
    """
    Test all implemented filters on the given image.
    
    Args:
        image_path: Path to the test image
        save_path: Path to save the result image (if None, display instead)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply filters
    print("Applying mean filter...")
    mean3x3 = mean_filter(image_rgb, kernel_size=3)
    mean5x5 = mean_filter(image_rgb, kernel_size=5)
    
    print("Applying Gaussian filter...")
    gauss3x3 = gaussian_filter(image_rgb, kernel_size=3, sigma=1.0)
    gauss5x5 = gaussian_filter(image_rgb, kernel_size=5, sigma=1.5)

    print("Applying Laplacian filter...")
    laplacian_std = laplacian_filter(gray, kernel_type='standard')
    laplacian_diag = laplacian_filter(gray, kernel_type='diagonal')
    
    print("Applying Sobel filter...")
    sobel_x = sobel_filter(gray, direction='x')
    sobel_y = sobel_filter(gray, direction='y')
    sobel_mag, sobel_dir = sobel_filter(gray, direction='both')
    
    # Normalize edge detection results for display
    laplacian_std_norm = normalize_image(laplacian_std)
    laplacian_diag_norm = normalize_image(laplacian_diag)
    sobel_x_norm = normalize_image(sobel_x)
    sobel_y_norm = normalize_image(sobel_y)
    sobel_mag_norm = normalize_image(sobel_mag)
    
    # Add noise and test denoising
    print("Testing noise reduction...")
    noisy_gray = add_noise(gray, noise_type='gaussian', var=0.02)
    noisy_saltpepper = add_noise(gray, noise_type='salt_pepper', var=0.02)

    denoised_mean = mean_filter(noisy_gray, kernel_size=5)
    denoised_gauss = gaussian_filter(noisy_gray, kernel_size=5, sigma=1.5)
    denoised_gauss2 = gaussian_filter(noisy_saltpepper, kernel_size=5, sigma=1.5)
    
    # Display or save results
    fig = plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 4, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(mean3x3.astype(np.uint8))
    plt.title('Mean Filter 3x3')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(mean5x5.astype(np.uint8))
    plt.title('Mean Filter 5x5')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(gauss3x3.astype(np.uint8))
    plt.title('Gaussian Filter 3x3')
    plt.axis('off')
    
    plt.subplot(3, 4, 5)
    plt.imshow(gauss5x5.astype(np.uint8))
    plt.title('Gaussian Filter 5x5')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(laplacian_std_norm, cmap='gray')
    plt.title('Laplacian (Standard)')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(laplacian_diag_norm, cmap='gray')
    plt.title('Laplacian (Diagonal)')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(sobel_x_norm, cmap='gray')
    plt.title('Sobel (X direction)')
    plt.axis('off')
    
    plt.subplot(3, 4, 9)
    plt.imshow(sobel_y_norm, cmap='gray')
    plt.title('Sobel (Y direction)')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(sobel_mag_norm, cmap='gray')
    plt.title('Sobel (Magnitude)')
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    plt.imshow(sobel_dir, cmap='hsv')
    plt.title('Sobel (Direction)')
    plt.axis('off')
    
    plt.subplot(3, 4, 12)
    plt.imshow(np.hstack([noisy_gray, denoised_gauss]), cmap='gray')
    plt.title('Noisy vs Denoised (Gaussian)')
    plt.axis('off')

    plt.subplot(3, 4, 12)
    plt.imshow(np.hstack([noisy_saltpepper, denoised_gauss2]), cmap='gray')
    plt.title('Noisy vs Denoised (Salt Pepper)')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Results saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def compare_with_opencv(image_path: str, save_path: str = None):
    """
    Compare implemented filters with OpenCV implementations.
    
    Args:
        image_path: Path to the test image
        save_path: Path to save the result image (if None, display instead)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply our filters
    print("Applying our filters...")
    our_mean = mean_filter(image_rgb, kernel_size=5)
    our_gauss = gaussian_filter(image_rgb, kernel_size=5, sigma=1.5)
    our_laplacian = laplacian_filter(gray, kernel_type='standard')
    our_sobel_x = sobel_filter(gray, direction='x', kernel_size=3)
    our_sobel_y = sobel_filter(gray, direction='y', kernel_size=3)
    
    # Apply OpenCV filters
    print("Applying OpenCV filters...")
    cv_mean = cv2.blur(image, (5, 5))
    cv_gauss = cv2.GaussianBlur(image, (5, 5), 1.5)
    cv_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    cv_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    cv_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Normalize for display
    our_laplacian_norm = normalize_image(our_laplacian)
    our_sobel_x_norm = normalize_image(our_sobel_x)
    our_sobel_y_norm = normalize_image(our_sobel_y)
    
    cv_laplacian_norm = normalize_image(cv_laplacian)
    cv_sobel_x_norm = normalize_image(cv_sobel_x)
    cv_sobel_y_norm = normalize_image(cv_sobel_y)

    # print("my laplacian", our_laplacian_norm[0])
    # print("cv laplacian", cv_laplacian_norm[0])
    
    # Display or save comparison
    fig = plt.figure(figsize=(15, 12))
    
    # Original image
    plt.subplot(3, 3, 2)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Mean filter comparison
    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(cv_mean, cv2.COLOR_BGR2RGB))
    plt.title('OpenCV Mean Filter')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(our_mean.astype(np.uint8))
    plt.title('Our Mean Filter')
    plt.axis('off')
    
    # Gaussian filter comparison
    plt.subplot(3, 3, 7)
    plt.imshow(cv2.cvtColor(cv_gauss, cv2.COLOR_BGR2RGB))
    plt.title('OpenCV Gaussian Filter')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(our_gauss.astype(np.uint8))
    plt.title('Our Gaussian Filter')
    plt.axis('off')
    
    # Create a separate figure for edge detection comparisons
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_smoothing.png'), dpi=300)
        print(f"Smoothing comparison saved to {save_path.replace('.png', '_smoothing.png')}")
    else:
        plt.show()
    
    plt.close(fig)
    
    # Edge detection comparison
    fig = plt.figure(figsize=(15, 12))
    
    # Laplacian comparison
    plt.subplot(3, 3, 1)
    plt.imshow(cv_laplacian_norm, cmap='gray')
    plt.title('OpenCV Laplacian')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(our_laplacian_norm, cmap='gray')
    plt.title('Our Laplacian')
    plt.axis('off')
    
    # Sobel X comparison
    plt.subplot(3, 3, 4)
    plt.imshow(cv_sobel_x_norm, cmap='gray')
    plt.title('OpenCV Sobel X')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(our_sobel_x_norm, cmap='gray')
    plt.title('Our Sobel X')
    plt.axis('off')
    
    # Sobel Y comparison
    plt.subplot(3, 3, 7)
    plt.imshow(cv_sobel_y_norm, cmap='gray')
    plt.title('OpenCV Sobel Y')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(our_sobel_y_norm, cmap='gray')
    plt.title('Our Sobel Y')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_edges.png'), dpi=300)
        print(f"Edge detection comparison saved to {save_path.replace('.png', '_edges.png')}")
    else:
        plt.show()
    
    plt.close(fig)


def test_canny(image_path: str, save_path: str = None):
    """
    Test Canny edge detector implementation and compare with OpenCV.
    
    Args:
        image_path: Path to the test image
        save_path: Path to save the result image (if None, display instead)
    """
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Test different parameter combinations
    low_thresholds = [0.05, 0.1, 0.15]
    high_thresholds = [0.15, 0.2, 0.3]
    sigmas = [1.0, 1.5, 2.0]
    
    # Create a figure to display results
    rows = len(low_thresholds)
    cols = len(sigmas) + 1  # +1 for original image in first column
    
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    
    # Display original image
    plt.subplot(rows, cols, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Test each parameter combination
    for i, low_thresh in enumerate(low_thresholds):
        high_thresh = high_thresholds[i]
        for j, sigma in enumerate(sigmas):
            print(f"Testing Canny with low_thresh={low_thresh}, high_thresh={high_thresh}, sigma={sigma}")
            
            # Apply our Canny edge detector
            start_time = time.time()
            our_canny = canny_edge_detector(
                image, 
                low_threshold=low_thresh,
                high_threshold=high_thresh,
                sigma=sigma
            )
            our_time = time.time() - start_time
            
            # Apply OpenCV's Canny for comparison
            start_time = time.time()
            # Convert thresholds to absolute values for OpenCV
            low_thresh_abs = int(low_thresh * 255)
            high_thresh_abs = int(high_thresh * 255)
            # Blur first as OpenCV's Canny does
            blurred = cv2.GaussianBlur(image, (5, 5), sigma)
            cv_canny = cv2.Canny(blurred, low_thresh_abs, high_thresh_abs)
            cv_time = time.time() - start_time
            
            # Plot the results
            plt.subplot(rows, cols, i * cols + j + 2)
            
            # Create a comparison image: left half our implementation, right half OpenCV
            height, width = image.shape
            comparison = np.zeros((height, width), dtype=np.uint8)
            comparison[:, :width//2] = our_canny[:, :width//2]
            comparison[:, width//2:] = cv_canny[:, width//2:]
            
            plt.imshow(comparison, cmap='gray')
            plt.title(f'L={low_thresh}, H={high_thresh}, σ={sigma}\n'
                     f'Ours: {our_time:.3f}s | CV: {cv_time:.3f}s')
            plt.axis('off')
            
            # Draw a vertical line to separate the two implementations
            plt.axvline(x=width//2, color='r', linestyle='-', linewidth=1)
            
            # Add labels to indicate which side is which
            plt.text(width//4, height - 20, 'Our Implementation', 
                    horizontalalignment='center', color='white', fontsize=10)
            plt.text(3*width//4, height - 20, 'OpenCV', 
                    horizontalalignment='center', color='white', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Canny comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function for creating the first set of figures
def create_figure_1(image_path: str, save_path: str = None):
    """
    Create the first figure with various filters applied to the image.
    
    Args:
        image_path: Path to the test image
        save_path: Path to save the result image (if None, display instead)
    """
    # Read image
    image = cv2.imread(image_path)
    image = image.copy().astype(np.float32)

    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Mean filter with kernel sizes 3, 5, 7
    mean3x3 = mean_filter(image_rgb, kernel_size=3)
    mean5x5 = mean_filter(image_rgb, kernel_size=5)
    mean7x7 = mean_filter(image_rgb, kernel_size=7)
    
    # Apply Gaussian filter with different sigma values
    gauss0 = gaussian_filter(image_rgb, kernel_size=3, sigma=0.01)
    gauss085 = gaussian_filter(image_rgb, kernel_size=3, sigma=0.85)
    gauss10 = gaussian_filter(image_rgb, kernel_size=3, sigma=10)
    
    # Add Gaussian and Salt-Pepper noise
    noisy_gaussian = add_noise(gray, noise_type='gaussian', var=0.02)
    noisy_saltpepper = add_noise(gray, noise_type='salt_pepper', var=0.02)
    
    # Apply denoising using Mean and Gaussian filters
    denoised_mean = mean_filter(noisy_gaussian, kernel_size=5)
    denoised_gauss = gaussian_filter(noisy_gaussian, kernel_size=5, sigma=1.5)
    denoised_gauss2 = gaussian_filter(noisy_saltpepper, kernel_size=5, sigma=1.5)
    
    # Apply filters with different padding
    mean_const_padding = mean_filter(image_rgb, kernel_size=3, padding_mode="constant")
    mean_reflect_padding = mean_filter(image_rgb,kernel_size=3,padding_mode="reflect")
    mean_wrap_padding = mean_filter(image_rgb,kernel_size=3,padding_mode="wrap") 
    
    # Create the first figure with captions
    fig = plt.figure(figsize=(18, 15))
    
    # Mean Filter with Kernel sizes 3, 5, 7
    plt.subplot(4, 5, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(4, 5, 2)
    plt.imshow(mean3x3.astype(np.uint8))
    plt.title('Mean Filter 3x3')
    plt.axis('off')
    
    plt.subplot(4, 5, 3)
    plt.imshow(mean5x5.astype(np.uint8))
    plt.title('Mean Filter 5x5')
    plt.axis('off')
    
    plt.subplot(4, 5, 4)
    plt.imshow(mean7x7.astype(np.uint8))
    plt.title('Mean Filter 7x7')
    plt.axis('off')
    
    # Gaussian Filter with different sigma values
    plt.subplot(4, 5, 5)
    plt.imshow(gauss0.astype(np.uint8))
    plt.title('Gaussian Filter (σ=0)')
    plt.axis('off')
    
    plt.subplot(4, 5, 6)
    plt.imshow(gauss085.astype(np.uint8))
    plt.title('Gaussian Filter (σ=0.85)')
    plt.axis('off')
    
    plt.subplot(4, 5, 7)
    plt.imshow(gauss10.astype(np.uint8))
    plt.title('Gaussian Filter (σ=10)')
    plt.axis('off')
    
    # Noisy Image and Denoised Images
    plt.subplot(4, 5, 8)
    plt.imshow(np.hstack([noisy_gaussian, denoised_gauss]), cmap='gray')
    plt.title('Noisy vs Denoised (Gaussian)')
    plt.axis('off')
    
    plt.subplot(4, 5, 9)
    plt.imshow(np.hstack([noisy_saltpepper, denoised_gauss2]), cmap='gray')
    plt.title('Noisy vs Denoised (Salt & Pepper)')
    plt.axis('off')
    
    # Filters with Padding Types
    plt.subplot(4, 5, 10)
    plt.imshow(mean_const_padding.astype(np.uint8))
    plt.title('Mean Filter (Constant Padding)')
    plt.axis('off')
    
    plt.subplot(4, 5, 11)
    plt.imshow(mean_reflect_padding.astype(np.uint8))
    plt.title('Mean Filter (Reflect Padding)')
    plt.axis('off')
    
    plt.subplot(4, 5, 12)
    plt.imshow(mean_wrap_padding.astype(np.uint8))
    plt.title('Mean Filter (Wrap Padding)')
    plt.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Results saved to {save_path}")
    else:
        plt.show()

# Function for creating the second set of figures
def create_figure_2(image_path: str, save_path: str = None):
    """
    Create the second figure with Sobel and Laplacian filters applied to the image.
    
    Args:
        image_path: Path to the test image
        save_path: Path to save the result image (if None, display instead)
    """
    # Read image
    image = cv2.imread(image_path)
    image = image.copy().astype(np.float32)

    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Sobel filter with different kernel sizes (3, 5, 7)
    sobel3x3, _ = sobel_filter(gray, direction='both', kernel_size=3)
    sobel_mag, _ = sobel_filter(gray, direction='both')
    sobel5x5, _ = sobel_filter(gray, direction='both', kernel_size=5)
    sobel7x7, _ = sobel_filter(gray, direction='both', kernel_size=7)
    
    # Add noise to the image
    noisy_gaussian = add_noise(gray, noise_type='gaussian', var=0.02)
    noisy_saltpepper = add_noise(gray, noise_type='salt_pepper', var=0.02)
    
    # Apply Laplacian filter and Sobel filter to noisy images
    laplacian_std = laplacian_filter(noisy_gaussian, kernel_type='standard')
    laplacian_diag = laplacian_filter(noisy_gaussian, kernel_type='diagonal')
    
    # Apply Sobel filter to noisy images
    sobel_noisy_gaussian, _ = sobel_filter(noisy_gaussian, direction='both')
    sobel_noisy_saltpepper, _ = sobel_filter(noisy_saltpepper, direction='both')
    
    # Filters with different padding for Sobel
    sobel_const_padding, _ = sobel_filter(gray, 'both', 3, 'constant')
    sobel_reflect_padding, _ = sobel_filter(gray, 'both', 3, 'reflect')
    sobel_wrap_padding, _ = sobel_filter(gray, 'both', 3, 'wrap')
    
    # Create the second figure with captions
    fig = plt.figure(figsize=(18, 15))
    
    # Sobel filter with kernel sizes 3, 5, 7
    plt.subplot(4, 5, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(4, 5, 2)
    plt.imshow(sobel3x3, cmap='gray')
    plt.title('Sobel Filter (3x3)')
    plt.axis('off')
    
    plt.subplot(4, 5, 3)
    plt.imshow(sobel5x5, cmap='gray')
    plt.title('Sobel Filter (5x5)')
    plt.axis('off')
    
    plt.subplot(4, 5, 4)
    plt.imshow(sobel7x7, cmap='gray')
    plt.title('Sobel Filter (7x7)')
    plt.axis('off')
    
    # Noisy Image and Denoised Images (Sobel and Laplacian)
    plt.subplot(4, 5, 5)
    plt.imshow(np.hstack([noisy_gaussian, laplacian_std]), cmap='gray')
    plt.title('Noisy vs Laplacian (Gaussian Noise)')
    plt.axis('off')
    
    plt.subplot(4, 5, 6)
    plt.imshow(np.hstack([noisy_saltpepper, laplacian_diag]), cmap='gray')
    plt.title('Noisy vs Laplacian (Salt & Pepper)')
    plt.axis('off')
    
    # Noisy Image and Filtered Images (Sobel)
    plt.subplot(4, 5, 7)
    plt.imshow(np.hstack([noisy_gaussian, sobel_noisy_gaussian]), cmap='gray')
    plt.title('Noisy vs Sobel (Gaussian Noise)')
    plt.axis('off')
    
    plt.subplot(4, 5, 8)
    plt.imshow(np.hstack([noisy_saltpepper, sobel_noisy_saltpepper]), cmap='gray')
    plt.title('Noisy vs Sobel (Salt & Pepper)')
    plt.axis('off')
    
    # Filters with Padding Types
    plt.subplot(4, 5, 9)
    plt.imshow(sobel_const_padding, cmap='gray')
    plt.title('Sobel Filter (Constant Padding)')
    plt.axis('off')
    
    plt.subplot(4, 5, 10)
    plt.imshow(sobel_reflect_padding, cmap='gray')
    plt.title('Sobel Filter (Reflect Padding)')
    plt.axis('off')
    
    plt.subplot(4, 5, 11)
    plt.imshow(sobel_wrap_padding, cmap='gray')
    plt.title('Sobel Filter (Wrap Padding)')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Results saved to {save_path}")
    else:
        plt.show()


def test_all(image_path: str, output_dir: str = "assignment1/filter_results"):
    """
    Run all tests on the given image.
    
    Args:
        image_path: Path to the test image
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Run basic filter tests
    test_basic_filters(
        image_path, 
        save_path=os.path.join(output_dir, f"{base_name}_basic_filters.png")
    )
    
    # Run comparison with OpenCV
    compare_with_opencv(
        image_path,
        save_path=os.path.join(output_dir, f"{base_name}_opencv_comparison.png")
    )
    
    # Run Canny edge detector tests
    test_canny(
        image_path,
        save_path=os.path.join(output_dir, f"{base_name}_canny_comparison.png")
    )
    
    create_figure_1(
        image_path,
        save_path=os.path.join(output_dir, f"{base_name}_fig1.png")
    )
    
    create_figure_2(image_path,
        save_path=os.path.join(output_dir, f"{base_name}_fig2.png")
    )
    
    print(f"All tests completed. Results saved to {output_dir}")

if __name__ == "__main__":
    # Replace with the path to your test image
    #test_image_path = "assignment1/example_images/test.jpg"
    
    # Run all tests
    #test_all(test_image_path)

    test_image_paths = [
    # "assignment1/example_images/test.jpg",
    # "assignment1/example_images/100px.png",
    "assignment1/example_images/self_photo.jpg",
    # "assignment1/example_images/500px.jpg",
    # "assignment1/example_images/Bliss_(Windows_XP).png",
    # "assignment1/example_images/download.jpeg",
    # "assignment1/example_images/Rainbow-Checkerboard-23.jpg",
    ]

    for path in test_image_paths:
        print(f"\n Running tests on: {path}")
        try:
            test_all(path)
        except TypeError as e:
            print(f"Skipping {path} due to TypeError: {e}")
        except Exception as e:
            print(f"Unexpected error on {path}: {e}")