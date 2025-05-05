"""
Loss functions for Siamese Networks
"""

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    
    Args:
        margin (float): The margin in the contrastive loss equation.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        """
        TODO: Implement the contrastive loss function
        
        The contrastive loss is defined as:
        L = (1-Y) * 1/2 * (D)^2 + Y * 1/2 * {max(0, margin - D)}^2
        
        where D is the euclidean distance between the outputs.
        """
        # Calculate euclidean distance
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        
        # Calculate contrastive loss
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + 
                                     (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = nn.functional.pairwise_distance(anchor, positive)
        neg_dist = nn.functional.pairwise_distance(anchor, negative)
        losses = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return losses.mean()