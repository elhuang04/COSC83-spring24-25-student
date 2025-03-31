import torch
import torch.nn.functional as F
import math

def calculate_psnr(sr_tensor, hr_tensor, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        sr_tensor (torch.Tensor): Super-resolution image
        hr_tensor (torch.Tensor): High-resolution ground truth
        max_val (float): Maximum value of the signal
    
    Returns:
        float: PSNR value in dB
    """
    # Ensure that tensors have the same shape
    if sr_tensor.shape != hr_tensor.shape:
        raise ValueError(f"Tensor shapes don't match: {sr_tensor.shape} vs {hr_tensor.shape}")
    
    # Ensure tensors are on the same device
    if sr_tensor.device != hr_tensor.device:
        hr_tensor = hr_tensor.to(sr_tensor.device)
    
    # Convert to float for calculations
    sr_tensor = sr_tensor.float()
    hr_tensor = hr_tensor.float()
    
    # Calculate Mean Squared Error
    mse = F.mse_loss(sr_tensor, hr_tensor)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * math.log10(max_val / math.sqrt(mse.item()))
    
    return psnr

def gaussian_kernel(size=11, sigma=1.5, channels=1, device='cpu'):
    """
    Create a Gaussian kernel for SSIM calculation
    
    Args:
        size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation of the Gaussian kernel
        channels (int): Number of channels
        device (str): Device to create the kernel on
    
    Returns:
        torch.Tensor: Gaussian kernel
    """
    # Create a 1D Gaussian kernel
    x = torch.arange(size, device=device) - (size - 1) / 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create a 2D Gaussian kernel
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    
    # Reshape for convolution
    kernel_2d = kernel_2d.view(1, 1, size, size)
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
    
    return kernel_2d

def calculate_ssim(sr_tensor, hr_tensor, window_size=11, max_val=1.0):
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        sr_tensor (torch.Tensor): Super-resolution image
        hr_tensor (torch.Tensor): High-resolution ground truth
        window_size (int): Size of the Gaussian window
        max_val (float): Maximum value of the signal
    
    Returns:
        float: SSIM value between -1 and 1
    """
    # Ensure that tensors have the same shape
    if sr_tensor.shape != hr_tensor.shape:
        raise ValueError(f"Tensor shapes don't match: {sr_tensor.shape} vs {hr_tensor.shape}")
    
    # If tensors are already on CUDA, use that device
    device = sr_tensor.device
    
    # Get dimensions
    channels = sr_tensor.size(0)
    
    # Convert to float for calculations
    sr_tensor = sr_tensor.float()
    hr_tensor = hr_tensor.float()
    
    # Constants for stability
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Create Gaussian kernel
    window = gaussian_kernel(size=window_size, sigma=1.5, channels=channels, device=device)
    
    # Pad inputs if necessary
    pad = window_size // 2
    
    # Compute means
    mu1 = F.conv2d(sr_tensor.unsqueeze(0), window, padding=pad, groups=channels).squeeze(0)
    mu2 = F.conv2d(hr_tensor.unsqueeze(0), window, padding=pad, groups=channels).squeeze(0)
    
    # Compute squares
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(sr_tensor.unsqueeze(0) ** 2, window, padding=pad, groups=channels).squeeze(0) - mu1_sq
    sigma2_sq = F.conv2d(hr_tensor.unsqueeze(0) ** 2, window, padding=pad, groups=channels).squeeze(0) - mu2_sq
    sigma12 = F.conv2d(sr_tensor.unsqueeze(0) * hr_tensor.unsqueeze(0), window, padding=pad, groups=channels).squeeze(0) - mu1_mu2
    
    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    
    # Return mean SSIM
    return ssim_map.mean().item()


# Simplified versions for faster computation (especially for validation during training)
def fast_ssim(sr_tensor, hr_tensor, max_val=1.0):
    """
    Simplified version of SSIM for faster computation
    Can be used during training for validation
    """
    if sr_tensor.dim() == 4:  # Batch of images
        batch_size = sr_tensor.size(0)
        ssim_values = []
        for i in range(batch_size):
            ssim_values.append(calculate_ssim(sr_tensor[i], hr_tensor[i], max_val=max_val))
        return sum(ssim_values) / len(ssim_values)
    else:  # Single image
        return calculate_ssim(sr_tensor, hr_tensor, max_val=max_val)

def fast_psnr(sr_tensor, hr_tensor, max_val=1.0):
    """
    Simplified version of PSNR for faster computation
    Can be used during training for validation
    """
    if sr_tensor.dim() == 4:  # Batch of images
        batch_size = sr_tensor.size(0)
        psnr_values = []
        for i in range(batch_size):
            psnr_values.append(calculate_psnr(sr_tensor[i], hr_tensor[i], max_val=max_val))
        return sum(psnr_values) / len(psnr_values)
    else:  # Single image
        return calculate_psnr(sr_tensor, hr_tensor, max_val=max_val)