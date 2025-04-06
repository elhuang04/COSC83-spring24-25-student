"""
Model architecture for Siamese Neural Network
"""
import torch
import torch.nn as nn
import torchvision

class Flatten(nn.Module):
    """Flatten layer to convert 4D tensor to 2D tensor."""
    def forward(self, input):
        return input.view(input.size(0), -1)

class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network implementation
    
    Args:
        contra_loss (bool): Whether to use contrastive loss (True) or BCE loss (False)
    """
    def __init__(self, contra_loss=False):
        super(SiameseNetwork, self).__init__()
        self.contra_loss = contra_loss
        
        # TODO: Initialize the ResNet18 backbone
        # Hint: Use torchvision.models.resnet18 and modify it appropriately
        # 1. Initialize the ResNet18 model
        # 2. Modify the first convolutional layer if needed
        # 3. Store the number of features from the final layer
        # 4. Remove the final classification layer
        
        # TODO: Create additional layers for BCE loss
        # Hint: You need fully connected layers to process the concatenated features
        # and a sigmoid activation for the final output
        
        # TODO: Initialize the weights of your network
        # Hint: Create a method to initialize weights and apply it to your layers
    
    def init_weights(self, m):
        # TODO: Implement weight initialization for linear layers
        # Hint: Use Xavier initialization for weights and small constant for biases
        pass
    
    def forward_once(self, x):
        """
        Forward pass for one input image
        """
        # TODO: Implement the forward pass for a single image
        # The function should return the feature vector for the input image
        # Hint: Pass the input through the backbone network and flatten the output
        pass
    
    def forward(self, input1, input2):
        """
        Forward pass for the Siamese network
        """
        # TODO: Implement the complete forward pass
        # 1. Get embeddings for both input images using forward_once
        # 2. Handle both cases: contrastive loss and BCE loss
        #    - For contrastive loss: return both embeddings
        #    - For BCE loss: concatenate embeddings, pass through FC layers, apply sigmoid
        pass