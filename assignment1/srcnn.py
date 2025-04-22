# Elizabeth Huang
# Last Modified: April 21, 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#5%
class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # You need to create:
        # 1. Two convolutional layers with kernel size 3, padding 1, and the same number of channels
        # 2. Two batch normalization layers
        # 3. ReLU activation
        #extracting feature, makes the image smaller
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)

        
    def forward(self, x):
        # 1. Store the input as the residual
        # 2. Pass the input through the first conv -> batch norm -> ReLU sequence
        # 3. Pass the result through the second conv -> batch norm sequence
        # 4. Add the residual to implement the skip connection
        # 5. Apply ReLU and return the result
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        return self.relu(out)


#5%
class UpscaleBlock(nn.Module):
    """Upscale block using sub-pixel convolution"""
    def __init__(self, in_channels, scale_factor):
        super(UpscaleBlock, self).__init__()
        # 1. Calculate output channels for sub-pixel convolution (hint: multiply in_channels by scale_factor^2)
        # 2. Create a convolutional layer with kernel size 3 and padding 1
        # 3. Create a pixel shuffle layer with the given scale factor
        # 4. Create a ReLU activation 
        # Calculate the number of output channels after sub-pixel convolution
        out_channels = in_channels * (scale_factor ** 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)

        
    def forward(self, x):
        # 1. Apply the convolutional layer
        # 2. Apply the pixel shuffle operation
        # 3. Apply ReLU and return the result
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.relu(x)

#10%
class SuperResolutionCNN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, num_features=64, num_blocks=16):
        """
        SuperResolution CNN with residual blocks and sub-pixel convolution
        
        Args:
            scale_factor (int): Upscaling factor
            num_channels (int): Number of input/output channels
            num_features (int): Number of feature channels
            num_blocks (int): Number of residual blocks
        """
        super(SuperResolutionCNN, self).__init__()
        self.scale_factor = scale_factor
        
        # 1. Create an initial convolution layer with kernel size 9, padding 4, followed by ReLU
        # 2. Create a sequence of residual blocks (use the ResidualBlock class)
        # 3. Create a mid convolution layer with kernel size 3, padding 1, followed by batch norm
        # 4. Create upscaling layers based on the scale factor:
        #    - For scale factors 2, 4, and 8 (powers of 2), use multiple x2 upscaling blocks
        #    - For scale factor 3, use a single x3 upscaling block
        #    - Raise an error for other scale factors
        # 5. Create a final convolution layer with kernel size 9, padding 4
        # 6. Initialize the weights using the _initialize_weights method
        
        # initial convolution layer
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)
        #residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        #mid convolution layer
        self.conv_mid = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_features)

        if (scale_factor & (scale_factor-1) == 0) and scale_factor != 0:
            up_blocks = []
            for i in range(int(math.log2(scale_factor))):
                up_blocks.append(UpscaleBlock(num_features,2))
            self.upscale = nn.Sequential(*up_blocks)
        
        elif scale_factor == 3:
            self.upscale = nn.Sequential(
                UpscaleBlock(num_features, 3),
            )
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")
        
        self.conv_out = nn.Conv2d(num_features, num_channels, kernel_size=9, padding=4)
        self._initialize_weights()
        
    def _initialize_weights(self):
        # For each module in the model:
        # 1. For convolutional layers, use Kaiming normal initialization for weights and zero initialization for biases
        # 2. For batch normalization layers, use ones for weights and zeros for biases
        # Initialize the weights for each module in the model
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        
    def forward(self, x):
        # 1. Apply the initial convolution and store the output for the global skip connection
        # 2. Pass the features through the residual blocks
        # 3. Apply the mid convolution
        # 4. Add the initial features (global residual learning)
        # 5. Apply the upscaling layers
        # 6. Apply the final convolution and return the result
        initial_features = self.relu(self.conv1(x))
        out = self.residual_blocks(initial_features)
        out = self.bn_mid(self.conv_mid(out))

        out = out + initial_features

        out = self.upscale(out)
        out = self.conv_out(out)
        return out
