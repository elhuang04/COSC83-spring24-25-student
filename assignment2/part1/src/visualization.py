import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def visualize_corners(image, corners, color=(0, 255, 0), radius=3, thickness=1):
    """
    Visualize Harris corners on the image.
    
    Args:
        image (numpy.ndarray): Input image
        corners (numpy.ndarray): Binary image with corners
        color (tuple): BGR color for corners
        radius (int): Radius of corner circles
        thickness (int): Thickness of corner circles
    
    Returns:
        numpy.ndarray: Image with visualized corners
    """
    # Make a copy of the image
    vis_image = image.copy()
    
    # Convert grayscale to color if needed
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Get corner coordinates
    y_coords, x_coords = np.where(corners)
    
    # Draw circles at corner positions
    for x, y in zip(x_coords, y_coords):
        cv2.circle(vis_image, (x, y), radius, color, thickness)
    
    return vis_image

def visualize_keypoints(image, keypoints, color=(0, 255, 0)):
    """
    Visualize keypoints on the image.
    
    Args:
        image (numpy.ndarray): Input image
        keypoints (list): List of cv2.KeyPoint objects
        color (tuple): BGR color for keypoints
    
    Returns:
        numpy.ndarray: Image with visualized keypoints
    """
    # Make a copy of the image
    vis_image = image.copy()
    
    # Convert grayscale to color if needed
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2BGR)
    
    # Draw keypoints with orientation and scale
    return cv2.drawKeypoints(
        vis_image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=color
    )

def visualize_matches(img1, keypoints1, img2, keypoints2, matches, 
                      inliers=None, color_inliers=(0, 255, 0), color_outliers=(0, 0, 255)):
    """
    Visualize matches between two images.
    
    Args:
        img1 (numpy.ndarray): First image
        keypoints1 (list): Keypoints in the first image
        img2 (numpy.ndarray): Second image
        keypoints2 (list): Keypoints in the second image
        matches (list): List of DMatch objects
        inliers (numpy.ndarray): Binary mask for inlier matches
        color_inliers (tuple): BGR color for inlier matches
        color_outliers (tuple): BGR color for outlier matches
    
    Returns:
        numpy.ndarray: Image with visualized matches
    """
    # Create output image
    img_out = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    
    # Convert grayscale to color if needed
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Copy images to output
    img_out[0:img1.shape[0], 0:img1.shape[1]] = img1
    img_out[0:img2.shape[0], img1.shape[1]:] = img2
    
    # Draw matches
    for i, match in enumerate(matches):
        # Get keypoint coordinates
        x1, y1 = int(keypoints1[match.queryIdx].pt[0]), int(keypoints1[match.queryIdx].pt[1])
        x2, y2 = int(keypoints2[match.trainIdx].pt[0]) + img1.shape[1], int(keypoints2[match.trainIdx].pt[1])
        
        # Determine color based on inlier status
        if inliers is not None and i < len(inliers):
            color = color_inliers if inliers[i] else color_outliers
        else:
            color = color_inliers
        
        # Draw line
        cv2.line(img_out, (x1, y1), (x2, y2), color, 1)
        # Draw circles at keypoints
        cv2.circle(img_out, (x1, y1), 4, color, 1)
        cv2.circle(img_out, (x2, y2), 4, color, 1)
    
    return img_out

def visualize_harris_response(response, colormap='viridis'):
    """
    Visualize Harris corner response.
    
    Args:
        response (numpy.ndarray): Harris corner response
        colormap (str): Matplotlib colormap name
    
    Returns:
        numpy.ndarray: Visualized response image (BGR)
    """
    # Normalize response
    if response.max() > 0:
        response_norm = response / response.max()
    else:
        response_norm = response
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    response_colored = cmap(response_norm)
    
    # Convert to BGR
    response_bgr = (response_colored[:, :, :3] * 255).astype(np.uint8)
    response_bgr = cv2.cvtColor(response_bgr, cv2.COLOR_RGB2BGR)
    
    return response_bgr

def create_match_ranking_visualization(image_pairs, quality_scores, n_best=5, n_worst=5):
    """
    Create visualization of the best and worst matching image pairs.
    
    Args:
        image_pairs (list): List of tuples (img1, img2, matches_img)
        quality_scores (list): List of quality scores for image pairs
        n_best (int): Number of best matches to show
        n_worst (int): Number of worst matches to show
    
    Returns:
        numpy.ndarray: Visualization of the best and worst matches
    """
    # Sort image pairs by quality score
    sorted_indices = np.argsort(quality_scores)
    
    # Get best and worst matches
    best_indices = sorted_indices[-n_best:][::-1]
    worst_indices = sorted_indices[:n_worst]
    
    # Create figure
    fig, axes = plt.subplots(
        n_best + n_worst, 1, 
        figsize=(12, 4 * (n_best + n_worst)),
        constrained_layout=True
    )
    
    # Plot best matches
    for i, idx in enumerate(best_indices):
        axes[i].imshow(cv2.cvtColor(image_pairs[idx][2], cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Best Match #{i+1}, Score: {quality_scores[idx]:.4f}")
        axes[i].axis('off')
    
    # Plot worst matches
    for i, idx in enumerate(worst_indices):
        axes[n_best + i].imshow(cv2.cvtColor(image_pairs[idx][2], cv2.COLOR_BGR2RGB))
        axes[n_best + i].set_title(f"Worst Match #{i+1}, Score: {quality_scores[idx]:.4f}")
        axes[n_best + i].axis('off')
    
    # Convert plot to image
    fig.canvas.draw()
    ranking_vis = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Convert to BGR
    ranking_vis = cv2.cvtColor(ranking_vis, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    
    return ranking_vis