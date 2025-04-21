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

        # First conv layer: maintains size due to padding=1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Second conv layer: same config as the first
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)

        
    def forward(self, x):
        # TODO: Implement the forward pass of the residual block
        # 1. Store the input as the residual
        # 2. Pass the input through the first conv -> batch norm -> ReLU sequence
        # 3. Pass the result through the second conv -> batch norm sequence
        # 4. Add the residual to implement the skip connection
        # 5. Apply ReLU and return the result

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        print("error here?????????????")
        out = self.conv2(out)
        out = self.bn2(out)
        print("no error here")

        out += residual
        return self.relu(out)


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
        # Calculate the number of output channels after sub-pixel convolution
        out_channels = in_channels * (scale_factor ** 2)
        # Convolution layer to increase channels for sub-pixel shuffle
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Pixel shuffle layer to upsample the spatial resolution
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

        
    def forward(self, x):
        # TODO: Implement the forward pass of the upscale block
        # 1. Apply the convolutional layer
        # 2. Apply the pixel shuffle operation
        # 3. Apply ReLU and return the result
        # Apply convolution
        x = self.conv(x)
        # Apply pixel shuffle (upsampling)
        x = self.pixel_shuffle(x)
        # Apply ReLU activation
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
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        # Mid convolution layer
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
        

        # Final convolution to restore the output channels
        self.conv_out = nn.Conv2d(num_features, num_channels, kernel_size=9, padding=4)
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        # TODO: Implement weight initialization
        # For each module in the model:
        # 1. For convolutional layers, use Kaiming normal initialization for weights and zero initialization for biases
        # 2. For batch normalization layers, use ones for weights and zeros for biases
        # Initialize the weights for each module in the model
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal initialization for weights of Conv2d layers
                nn.init.kaiming_normal_(m.weight)
                # Zero initialization for biases of Conv2d layers
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize batch norm weights to ones
                nn.init.ones_(m.weight)
                # Initialize batch norm biases to zeros
                nn.init.zeros_(m.bias)

        
    def forward(self, x):
        # TODO: Implement the forward pass of the Super Resolution CNN
        # 1. Apply the initial convolution and store the output for the global skip connection
        # 2. Pass the features through the residual blocks
        # 3. Apply the mid convolution
        # 4. Add the initial features (global residual learning)
        # 5. Apply the upscaling layers
        # 6. Apply the final convolution and return the result
        
        initial_features = self.relu(self.conv1(x))
        
        # Step 2: Pass the features through the residual blocks
        out = self.residual_blocks(initial_features)
        

        print(out.shape)
        
        # Step 3: Apply the mid convolution
        out = self.bn_mid(self.conv_mid(out))
        
        # Step 4: Add the initial features (global residual learning)
        out = out + initial_features
        
        # Step 5: Apply the upscaling layers
        out = self.upscale(out)
        
        # Step 6: Apply the final convolution and return the result
        out = self.conv_out(out)
        
        return out
