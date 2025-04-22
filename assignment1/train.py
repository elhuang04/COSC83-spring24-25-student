# Elizabeth Huang
# Last Modified: April 2, 2025

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Import your dataloader and model
from dataloader import get_dataloader
from srcnn import SuperResolutionCNN

# Metrics
from metrics import calculate_psnr, calculate_ssim, fast_psnr, fast_ssim

#10%
def train(config):
    """
    Train the SuperResolution model
    
    Args:
        config (dict): Configuration parameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directory for saving model and results
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    
    # Create the model
    model = SuperResolutionCNN(
        scale_factor=config['scale_factor'],
        num_channels=3,
        num_features=config['num_features'],
        num_blocks=config['num_blocks']
    ).to(device)
    
    # Load checkpoint if continuing training
    start_epoch = 0
    if config['resume'] and os.path.exists(config['resume']):
        checkpoint = torch.load(config['resume'])
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")
    
    # Loss function
    criterion = nn.L1Loss() #TODO: extra credit - L2 loss implementation
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_decay_step'],
        gamma=config['lr_decay_gamma']
    )
    
    # Create dataloaders
    train_dataloader = get_dataloader(
        hr_dir=config['train_dir'],
        batch_size=config['batch_size'],
        patch_size=config['patch_size'],
        fixed_scale=config['scale_factor'],
        downsample_methods=config['downsample_methods'],
        num_workers=config['num_workers']
    )
    
    val_dataloader = get_dataloader(
        hr_dir=config['val_dir'],
        batch_size=config['batch_size'],
        patch_size=config['patch_size'],
        fixed_scale=config['scale_factor'],
        downsample_methods=['bicubic'],  # Use only bicubic for validation (faster)
        num_workers=config['num_workers']
    )
    
    # Training statistics
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    best_psnr = 0.0
    # Your implementation should handle the following:
    #
    # 1. Create a loop that iterates for the specified number of epochs
    #    - Start from start_epoch (in case training is resumed)
    #    - Continue until config['num_epochs']
    #
    # 2. For each epoch, implement the training phase:
    #    - Set the model to training mode
    #    - Create a progress bar using tqdm
    #    - Iterate through the training batches
    #    - Move data to the device
    #    - Perform forward pass, loss calculation, backward pass, and optimization
    #    - Track and display the loss
    #
    # 3. Update the learning rate using the scheduler after each epoch
    #
    # 4. Implement validation at specified intervals:
    #    - Check if validation should be performed based on config['validation_interval']
    #    - Set the model to evaluation mode
    #    - Iterate through validation batches with torch.no_grad()
    #    - Calculate validation loss, PSNR, and SSIM
    #    - Save sample images showing low-res input, super-resolution output, and high-res target
    #
    # 5. Save checkpoints:
    #    - Save the best model based on PSNR
    #    - Save regular checkpoints based on config['save_every']
    #
    # 6. Track and print training statistics
    #
    # Refer to the docstrings of other functions and the provided configuration
    # to understand the expected behavior and parameters.
    
    # Your code here
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        epoch_losses = []
        
        # progress bar
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in train_pbar:
            lr_imgs = batch['lr'].to(device)
            hr_imgs = batch['hr'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            sr_imgs = model(lr_imgs)
            
            # Calculate loss
            loss = criterion(sr_imgs, hr_imgs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_losses.append(loss.item())
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average training loss
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        
        # Only perform validation at specified intervals
        should_validate = (
            config['validation_interval'] > 0 and 
            (epoch + 1) % config['validation_interval'] == 0
        ) or (epoch + 1 == config['num_epochs'])
        
        if should_validate:
            # Validation
            model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            val_ssim = 0.0
            val_count = 0
            
            with torch.no_grad():
                # Progress bar for validation (limit number of batches to process)
                val_batch_limit = config.get('val_batch_limit', float('inf'))
                val_batches = list(val_dataloader)
                
                # Shuffle and limit validation batches
                if len(val_batches) > val_batch_limit:
                    random.shuffle(val_batches)
                    val_batches = val_batches[:val_batch_limit]
                
                val_pbar = tqdm(val_batches, desc="Validation")
                
                for batch_idx, batch in enumerate(val_pbar):
                    # Get data
                    lr_imgs = batch['lr'].to(device)
                    hr_imgs = batch['hr'].to(device)
                    
                    # Forward pass
                    sr_imgs = model(lr_imgs)
                    
                    # Calculate metrics
                    val_loss += criterion(sr_imgs, hr_imgs).item()
                    
                    # Calculate PSNR and SSIM more efficiently (batch-wise)
                    sr_imgs_clamped = sr_imgs.clamp(0.0, 1.0)
                    psnr = fast_psnr(sr_imgs_clamped.cpu(), hr_imgs.cpu())
                    ssim = fast_ssim(sr_imgs_clamped.cpu(), hr_imgs.cpu())
                    
                    val_psnr += psnr
                    val_ssim += ssim
                    val_count += 1
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        "loss": f"{val_loss/val_count:.4f}",
                        "PSNR": f"{val_psnr/val_count:.2f}",
                        "SSIM": f"{val_ssim/val_count:.4f}"
                    })
                    
                    # Save sample images from the first batch
                    if batch_idx == 0:
                        for i in range(min(3, lr_imgs.size(0))):
                            # Get images
                            lr_img = lr_imgs[i].cpu()
                            sr_img = sr_imgs_clamped[i].cpu()
                            hr_img = hr_imgs[i].cpu()
                            
                            # Up-sample LR image to match SR dimensions for visualization
                            # Get dimensions
                            _, _, sr_h, sr_w = sr_img.unsqueeze(0).shape
                            
                            # Resize LR to match SR using interpolate
                            lr_img_upscaled = torch.nn.functional.interpolate(
                                lr_img.unsqueeze(0), 
                                size=(sr_h, sr_w), 
                                mode='bicubic', 
                                align_corners=False
                            ).squeeze(0).clamp(0, 1)
                            
                            save_image(
                                torch.cat([
                                    lr_img_upscaled,
                                    sr_img,
                                    hr_img
                                ], dim=2),
                                os.path.join(config['sample_dir'], f"epoch_{epoch+1}_sample_{i}.png")
                            )
            
            #calculate average validation metrics
            val_loss /= val_count
            val_psnr /= val_count
            val_ssim /= val_count
            
            # Save metrics
            val_losses.append(val_loss)
            val_psnrs.append(val_psnr)
            val_ssims.append(val_ssim)
            
            print(f"Epoch {epoch+1}/{config['num_epochs']} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'psnr': val_psnr,
                    'ssim': val_ssim,
                }, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
                print(f"Saved best model with PSNR: {val_psnr:.2f}")
        else:
            # Print training-only summary
            print(f"Epoch {epoch+1}/{config['num_epochs']} - Train Loss: {train_loss:.4f}")
            # placeholder values to keep arrays aligned
            val_losses.append(None)
            val_psnrs.append(None)
            val_ssims.append(None)
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss if should_validate else None,
                'psnr': val_psnr if should_validate else None,
                'ssim': val_ssim if should_validate else None,
            }, os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))

    #--------------------------------------------------------------------
    
    # Plot training history (after training loop completes)
    plt.figure(figsize=(12, 8))
    
    # Plot loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    # Filter out None values for validation
    val_epochs = [i for i, v in enumerate(val_losses) if v is not None]
    val_loss_values = [v for v in val_losses if v is not None]
    plt.plot(val_epochs, val_loss_values, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot PSNR
    plt.subplot(1, 3, 2)
    val_psnr_values = [v for v in val_psnrs if v is not None]
    plt.plot(val_epochs, val_psnr_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR on Validation Set')
    
    # Plot SSIM
    plt.subplot(1, 3, 3)
    val_ssim_values = [v for v in val_ssims if v is not None]
    plt.plot(val_epochs, val_ssim_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM on Validation Set')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['checkpoint_dir'], 'training_history.png'))
    
    print("Training completed!")


if __name__ == "__main__":
    # Configuration
    config = {
        # Model parameters
        'scale_factor': 4,              # Upscaling factor
        'num_features': 64,             # Number of feature channels
        'num_blocks': 16,               # Number of residual blocks
        
        # Data parameters
        'train_dir': 'DIV2K_train_HR',  # Training data directory
        'val_dir': 'DIV2K_valid_HR',    # Validation data directory
        'patch_size': 128,              # Size of high-resolution patches
        'downsample_methods': ['bicubic', 'bilinear', 'nearest', 'lanczos'],
        
        # Training parameters
        'batch_size': 16,                # Batch size
        'num_epochs': 10,               # Total number of epochs
        'learning_rate': 1e-4,           # Initial learning rate
        'lr_decay_step': 30,             # Epoch interval to decay LR
        'lr_decay_gamma': 0.5,           # Multiplicative factor of learning rate decay
        'num_workers': 4,                # Number of data loading workers
        'validation_interval': 1,        # Epoch interval to perform validation (set to 5 for faster training)
        'val_batch_limit': 10,           # Maximum number of validation batches to process
        
        # Checkpoint parameters
        'checkpoint_dir': 'checkpoints', # Directory to save checkpoints
        'sample_dir': 'samples',         # Directory to save sample images
        'save_every': 5,                 # Save checkpoint every N epochs
        'resume': None,                  # Path to checkpoint to resume from
    }
    
    train(config)