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
    def __init__(self, contra_loss=False, triplet_loss=False):
        super(SiameseNetwork, self).__init__()
        self.contra_loss = contra_loss
        self.triplet_loss = triplet_loss
        
        # TODO: Initialize the ResNet18 backbone
        # Hint: Use torchvision.models.resnet18 and modify it appropriately
        # 1. Initialize the ResNet18 model
        # 2. Modify the first convolutional layer if needed
        # 3. Store the number of features from the final layer
        # 4. Remove the final classification layer
        resnet = torchvision.models.resnet18(pretrained=True)
        #resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.embedding_dim = resnet.fc.in_features
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # TODO: Create additional layers for BCE loss
        # Hint: You need fully connected layers to process the concatenated features
        # and a sigmoid activation for the final output
        if not self.contra_loss and not self.triplet_loss:
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, 512),  # Assuming the input will be a pair of images
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        # TODO: Initialize the weights of your network
        # Hint: Create a method to initialize weights and apply it to your layers
        self.apply(self.init_weights)

    def init_weights(self, m):
        # TODO: Implement weight initialization for linear layers
        # Hint: Use Xavier initialization for weights and small constant for biases
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
    
    def forward_once(self, x):
        """
        Forward pass for one input image
        """
        # TODO: Implement the forward pass for a single image
        # The function should return the feature vector for the input image
        # Hint: Pass the input through the backbone network and flatten the output
        x = self.feature_extractor(x)      
        x = x.view(x.size(0), -1)         
        return x

    
    def forward(self, input1, input2):
        """
        Forward pass for the Siamese network
        """
        # TODO: Implement the complete forward pass
        # 1. Get embeddings for both input images using forward_once
        # 2. Handle both cases: contrastive loss and BCE loss
        #    - For contrastive loss: return both embeddings
        #    - For BCE loss: concatenate embeddings, pass through FC layers, apply sigmoid
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if self.contra_loss or self.triplet_loss:
            return output1, output2 
        else:
            combined = torch.cat((output1, output2), dim=1)
            similarity_score = self.classifier(combined)
            return similarity_score

    
    def forward_triplet(self, anchor, positive, negative):
        anchor_emb = self.forward_once(anchor)
        positive_emb = self.forward_once(positive)
        negative_emb = self.forward_once(negative)
        return anchor_emb, positive_emb, negative_emb
