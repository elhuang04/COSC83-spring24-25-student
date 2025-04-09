import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#5%
class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # TODO: Implement the residual block constructor
        # You need to create:
        # 1. Two convolutional layers with kernel size 3, padding 1, and the same number of channels
        # 2. Two batch normalization layers
        # 3. ReLU activation
        #extracting feature, makes the image smaller

        self.rel = F.relu()
        pass
        
    def forward(self, x):
        # TODO: Implement the forward pass of the residual block
        # 1. Store the input as the residual
        # 2. Pass the input through the first conv -> batch norm -> ReLU sequence
        # 3. Pass the result through the second conv -> batch norm sequence
        # 4. Add the residual to implement the skip connection
        # 5. Apply ReLU and return the result
        pass

#5%
class UpscaleBlock(nn.Module):
    """Upscale block using sub-pixel convolution"""
    def __init__(self, in_channels, scale_factor):
        super(UpscaleBlock, self).__init__()
        # TODO: Implement the upscale block constructor
        # 1. Calculate output channels for sub-pixel convolution (hint: multiply in_channels by scale_factor^2)
        # 2. Create a convolutional layer with kernel size 3 and padding 1
        # 3. Create a pixel shuffle layer with the given scale factor
        # 4. Create a ReLU activation 
        pass
        
    def forward(self, x):
        # TODO: Implement the forward pass of the upscale block
        # 1. Apply the convolutional layer
        # 2. Apply the pixel shuffle operation
        # 3. Apply ReLU and return the result
        pass

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
        
        # TODO: Implement the constructor for the Super Resolution CNN
        # 1. Create an initial convolution layer with kernel size 9, padding 4, followed by ReLU
        # 2. Create a sequence of residual blocks (use the ResidualBlock class)
        # 3. Create a mid convolution layer with kernel size 3, padding 1, followed by batch norm
        # 4. Create upscaling layers based on the scale factor:
        #    - For scale factors 2, 4, and 8 (powers of 2), use multiple x2 upscaling blocks
        #    - For scale factor 3, use a single x3 upscaling block
        #    - Raise an error for other scale factors
        # 5. Create a final convolution layer with kernel size 9, padding 4
        # 6. Initialize the weights using the _initialize_weights method
        pass
        
    def _initialize_weights(self):
        # TODO: Implement weight initialization
        # For each module in the model:
        # 1. For convolutional layers, use Kaiming normal initialization for weights and zero initialization for biases
        # 2. For batch normalization layers, use ones for weights and zeros for biases
        pass
        
    def forward(self, x):
        # TODO: Implement the forward pass of the Super Resolution CNN
        # 1. Apply the initial convolution and store the output for the global skip connection
        # 2. Pass the features through the residual blocks
        # 3. Apply the mid convolution
        # 4. Add the initial features (global residual learning)
        # 5. Apply the upscaling layers
        # 6. Apply the final convolution and return the result
        pass