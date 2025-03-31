# Computer Vision Assignment: Image Filtering and Super-Resolution

## Overview
This assignment consists of two parts:
1. Traditional image filtering techniques implemented from scratch
2. Deep learning-based image super-resolution

In the first part, you will implement several fundamental image filtering techniques from scratch using Python. By implementing these filters manually rather than using library functions, you will gain a deeper understanding of how these operations work at a mathematical and algorithmic level.

In the second part, you will work with deep learning-based image super-resolution techniques to explore how modern approaches can enhance image quality beyond traditional methods.

## Learning Objectives
- Understand the mathematical foundations of image filtering operations
- Implement convolution operations from scratch
- Apply different filtering techniques to solve specific image processing problems
- Analyze the effects of different filters on various types of images
- Compare the performance and results of different filtering approaches
- Understand modern deep learning approaches to image super-resolution
- Compare traditional filtering with learning-based methods

## Requirements

### Dependencies
- Python 3.7+
- NumPy
- Matplotlib (for visualization)
- OpenCV (for image reading/writing and comparison only)
- Pillow (alternative for image I/O)
- PyTorch(for the super-resolution part)

### Part 1: Traditional Filters Implementation
You must implement the following filters from scratch:

1. **Mean Filter (Box Filter)**
   - Implement a simple averaging filter with customizable kernel size
   - Handle border conditions appropriately

2. **Gaussian Filter**
   - Implement a Gaussian smoothing filter
   - Allow for parameterized sigma and kernel size
   - Generate the Gaussian kernel based on the formula

3. **Laplacian Filter**
   - Implement the Laplacian operator for edge detection
   - Compare different Laplacian kernel variants

4. **Sobel Filter**
   - Implement both horizontal and vertical Sobel operators
   - Combine them to produce gradient magnitude and direction
   - Implement adjustable filter sizes (3x3, 5x5, etc.)

### Part 2: Image Super-Resolution with Deep Learning
For the super-resolution section, you will implement a CNN-based model that can upscale low-resolution images to higher resolution. The focus is on understanding and implementing modern approaches to enhance image quality beyond traditional interpolation methods.

#### 1. CNN Architecture for Super-Resolution

You need to implement a Convolutional Neural Network for super-resolution with the following architecture:

- **Input Layer**: Takes low-resolution RGB images (3 channels)
- **Feature Extraction Stage**:
  - Initial convolution layer with large receptive field (9×9 kernel)
  - Multiple residual blocks (16 recommended) consisting of:
    - Conv (3×3) → BatchNorm → ReLU
    - Conv (3×3) → BatchNorm
    - Skip connection (element-wise addition)
  - Global skip connection around all residual blocks
  
- **Upscaling Stage**:
  - Sub-pixel convolution (PixelShuffle) for efficient upscaling
  - For 4× upscaling, use two 2× upscale blocks or one 4× block
  - Each upscale block should perform: Conv → PixelShuffle → ReLU
  
- **Reconstruction Stage**:
  - Final convolution layer (9×9 kernel) to produce the HR RGB output

The model should support various scale factors (2×, 3×, 4×) with appropriate adjustments to the upscaling stage. The architecture follows a global residual learning approach, where the network learns to predict the residual between the bicubic-interpolated LR image and the HR target.

#### 2. Training and Evaluation Process

For training the super-resolution model, you'll need to:

1. **Prepare the dataloader**: (download the dataset from https://data.vision.ee.ethz.ch/cvl/DIV2K/ using the High Resolution Images, please split the data yourself)
   - Create a custom dataset class that loads HR images
   - Implement random cropping of HR patches
   - Generate LR counterparts using random downsampling methods (bicubic, bilinear, etc.)
   - Ensure proper normalization and augmentation (flips, rotations)

2. **Implement the training loop**:
   - Fill in the TODOs in the training loop code provided
   - Setup appropriate loss function (L1 loss recommended)
   - Configure optimizer (Adam with learning rate ~1e-4)
   - Implement learning rate scheduling
   - Save checkpoints and sample images during training
   - Track and plot metrics (PSNR, SSIM) during validation

3. **Evaluate the model**:
   - Test on unseen images
   - Compare with traditional upscaling methods
   - Analyze both quantitative metrics and visual quality

#### 3. Performance Metrics
You should evaluate your model using:
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Visual comparison with bicubic interpolation

### Implementation Details

#### Part 1: Traditional Filters
Implement the following core functions:

1. **Convolution Function**
   ```python
   def convolve2d(image, kernel, padding_mode='constant'):
       """
       Apply 2D convolution operation on an image with a given kernel.
       
       Args:
           image: Input image (2D or 3D numpy array)
           kernel: Convolution kernel (2D numpy array)
           padding_mode: How to handle borders ('constant', 'reflect', 'replicate', etc.)
           
       Returns:
           Convolved image (same size as input)
       """
       # Your implementation here
   ```

2. **Mean Filter**
   ```python
   def mean_filter(image, kernel_size, padding_mode='constant'):
       """
       Apply mean filtering to an image.
       
       Args:
           image: Input image
           kernel_size: Size of the kernel (e.g., 3 for 3x3, 5 for 5x5)
           padding_mode: How to handle borders
           
       Returns:
           Filtered image
       """
       # Your implementation here
   ```

3. **Gaussian Filter**
   ```python
   def gaussian_kernel(size, sigma):
       """
       Generate a Gaussian kernel.
       
       Args:
           size: Kernel size (must be odd)
           sigma: Standard deviation of the Gaussian
           
       Returns:
           Gaussian kernel (normalized)
       """
       # Your implementation here
       
   def gaussian_filter(image, kernel_size, sigma, padding_mode='constant'):
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
       # Your implementation here
   ```

4. **Laplacian Filter**
   ```python
   def laplacian_filter(image, kernel_type='standard', padding_mode='constant'):
       """
       Apply Laplacian filtering for edge detection.
       
       Args:
           image: Input image
           kernel_type: Type of Laplacian kernel ('standard', 'diagonal', etc.)
           padding_mode: How to handle borders
           
       Returns:
           Filtered image
       """
       # Your implementation here
   ```

5. **Sobel Filter**
   ```python
   def sobel_filter(image, direction='both', kernel_size=3, padding_mode='constant'):
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
       # Your implementation here
   ```

#### Part 2: Super-Resolution CNN Implementation

For the Super-Resolution CNN model, you need to implement:

1. **Residual Block**
   ```python
   class ResidualBlock(nn.Module):
       """
       Residual block with skip connection.
       
       Structure:
       Conv(3×3) → BatchNorm → ReLU → Conv(3×3) → BatchNorm → Add input
       """
       # TODO: Implement the residual block
   ```

2. **Upscale Block**
   ```python
   class UpscaleBlock(nn.Module):
       """
       Upscale block using sub-pixel convolution (PixelShuffle).
       
       Structure:
       Conv → PixelShuffle → ReLU
       """
       # TODO: Implement the upscale block
   ```

3. **SuperResolutionCNN**
   ```python
   class SuperResolutionCNN(nn.Module):
       """
       Complete Super-Resolution CNN model with residual blocks and upscaling.
       
       Structure:
       1. Initial feature extraction
       2. Multiple residual blocks
       3. Global skip connection
       4. Upscaling blocks
       5. Final reconstruction
       """
       # TODO: Implement the full model
   ```

4. **Training Loop**
   - Fill in the TODOs in the training loop
   - Implement data loading, loss calculation, optimization, validation, and checkpoint saving

### Testing and Evaluation

For Part 1 (Traditional Filters), test your implementations on the following types of images:
1. A synthetic image with sharp edges (e.g., checkerboard or geometric shapes)
2. A natural image with various textures (e.g., landscape)
3. A portrait/face image
4. A noisy image (add Gaussian and salt-and-pepper noise to test denoising capabilities)

For each traditional filter and image, analyze:
- The effect of different parameter settings (kernel size, sigma values)
- The handling of image borders with different padding strategies
- The performance on noisy vs. clean images
- The differences between your implementation and OpenCV's built-in functions

For Part 2 (Super-Resolution), evaluate:
- PSNR and SSIM metrics on **your own** images *(should be more than 10)*
- Visual quality compared to bicubic interpolation
- Limitations and failure cases of the model

### Submission Requirements

Submit your assignment with the following components:

1. **Source Code**:
   - All implementation files with well-documented code
   - A main script that demonstrates all filters with example images

2. **Report**:
   - Must be prepared using LaTeX and submitted as PDF.
   - Must include your own personally captured images as demonstration data for both parts
   - Should include:
     * Brief introduction and methodology
     * Results section showing:
       - Part 1: Before/after images for each traditional filter implementation
       - Part 2: Training curves for the super-resolution model (loss vs. epochs)
       - Part 2: Visual comparison of super-resolution results vs. original images
     * Brief discussion of results and conclusion
   - Use proper LaTeX formatting with appropriate figures and captions

3. **Test Images**:
   - Include all test images used in your experiments
   - At least one image must be personally captured by you (not from the internet)

## Grading Criteria

Your assignment will be evaluated based on:

| Component | Weight | Description |
|-----------|--------|-------------|
| Implementation correctness | 80% | Correct implementation of all required filters |
| Report | 20% | LaTeX report with clear explanations, proper formatting, and comprehensive analysis |

## Bonus Points (up to 10%) (Choose one between a and b)
### a (5%)
For extra credit, implement the Canny edge detector algorithm from scratch, building on your Sobel filter implementation. This should include:

1. Noise reduction (using your Gaussian filter)
2. Gradient calculation (using your Sobel filter)
3. Non-maximum suppression
4. Double thresholding
5. Edge tracking by hysteresis

Your implementation should allow for parameter adjustment (threshold values) and demonstrate results on various test images.
### b (10%)
Change the training objective to L2 loss and Perception loss(Please check [this](https://deepai.org/machine-learning-glossary-and-terms/perceptual-loss-function) for the brief intro to it ). 
Compare the differences in the report.

## Timeline

- **Assigned**: [Insert Date]
- **Due**: [Insert Date] at 11:59 PM
- **Late submissions**: [Your policy]

## Resources

See the course website for additional resources and reference materials on image filtering.

---

### Honor Code

By submitting this assignment, you confirm that you have completed the implementation yourself without using external code for the core functionality. You may consult references for mathematical formulations but must implement the algorithms yourself.