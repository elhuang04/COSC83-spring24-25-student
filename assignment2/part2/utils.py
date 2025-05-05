"""
Utility functions for Siamese Networks
"""

import torch
import matplotlib.pyplot as plt


def threshold_sigmoid(t):
    """
    Convert probability to binary prediction (prob > 0.5 --> 1 else 0)
    
    Args:
        t: Tensor of probabilities
        
    Returns:
        Binary tensor with 1s where prob > 0.5 and 0s elsewhere
    """
    threshold = t.clone()
    threshold.data.fill_(0.5)
    return (t > threshold).float()


def threshold_contrastive_loss(input1, input2, m):
    """
    Convert distance to binary prediction (dist < m --> 1 else 0)
    
    Args:
        input1: First embedding tensor
        input2: Second embedding tensor
        m: Margin value
        
    Returns:
        Binary tensor with 0s where dist < m and 1s elsewhere
    """
    diff = input1 - input2
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    dist = torch.sqrt(dist_sq)
    threshold = dist.clone()
    threshold.data.fill_(m)
    return (dist < threshold).float().view(-1, 1)


def visualize_predictions(img1_set, img2_set, labels, predictions, n=5):
    """
    TODO: Implement visualization function to display pairs of faces and model predictions
    
    Args:
        img1_set: Batch of first images
        img2_set: Batch of second images
        labels: Ground truth labels (1 = different, 0 = same)
        predictions: Model predictions
        n: Number of pairs to display
    """
    plt.figure(figsize=(12, 4*n))
    
    for i in range(min(n, len(labels))):
        # Convert images back to display format
        img1 = img1_set[i].permute(1, 2, 0).cpu().numpy()
        img2 = img2_set[i].permute(1, 2, 0).cpu().numpy()
        
        # Normalize to [0, 1]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        
        # Get label and prediction
        label = labels[i].item()
        pred = predictions[i].item()
        
        # Plot
        plt.subplot(n, 2, 2*i+1)
        plt.imshow(img1)
        plt.title(f"Image 1")
        plt.axis('off')
        
        plt.subplot(n, 2, 2*i+2)
        plt.imshow(img2)
        plt.title(f"Image 2 (GT: {'Same' if label == 0 else 'Different'}, " + 
                  f"Pred: {'Same' if pred == 0 else 'Different'})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('pair_predictions.png')
    plt.close()

def threshold_triplet_loss(anchor, positive, negative, margin):
    """
    Apply a threshold-based criterion to triplet loss, ensuring that the anchor is closer to
    the positive example than the negative example by at least the margin.
    
    Args:
        anchor: The anchor embedding tensor
        positive: The positive embedding tensor
        negative: The negative embedding tensor
        margin: The margin value used to determine when the triplet is valid
    
    Returns:
        Binary tensor (1 = valid triplet, 0 = invalid triplet) based on the threshold condition
    """
    # Compute the pairwise distance between anchor-positive and anchor-negative pairs
    pos_dist = torch.norm(anchor - positive, p=2, dim=1)
    neg_dist = torch.norm(anchor - negative, p=2, dim=1)
    
    # Calculate the difference between the positive and negative distances
    dist_diff = pos_dist - neg_dist + margin
    
    # Apply threshold: If dist_diff > 0, we consider it a valid triplet
    # Return binary tensor: 1 for valid triplet, 0 for invalid triplet
    return (dist_diff < 0).float()
