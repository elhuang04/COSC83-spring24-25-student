import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Import your implementations
from src.harris import HarrisDetector
from src.descriptors import FeatureDescriptor, HarrisKeypointExtractor
from src.matching import FeatureMatcher, RANSAC

from utils.image_utils import load_image, resize_image

def test_harris():
    """Test Harris corner detector."""
    print("Testing Harris corner detector...")
    
    # Load test image
    img_path = "data/image_pairs/test_image.jpg"
    if not os.path.exists(img_path):
        print(f"Test image not found at {img_path}. Using placeholder...")
        # Create a simple test image if none exists
        img = np.zeros((300, 300), dtype=np.uint8)
        img[100:200, 100:200] = 255  # White square in the middle
        cv2.imwrite(img_path, img)
    
    img = load_image(img_path, grayscale=True)
    
    # Test Harris detector with different parameters
    thresholds = [0.01, 0.05, 0.1]
    k_values = [0.04, 0.05, 0.06]
    window_sizes = [3, 5, 7]
    
    # Create figure
    fig, axes = plt.subplots(len(thresholds), len(k_values), figsize=(15, 10))
    
    for i, threshold in enumerate(thresholds):
        for j, k in enumerate(k_values):
            # Create detector
            detector = HarrisDetector(k=k, threshold=threshold, window_size=3)
            
            # Detect corners
            corners, response = detector.detect_corners(img)
            
            # Get corner coordinates
            corner_coords = detector.get_corner_coordinates(corners)
            
            # Plot image and corners
            ax = axes[i, j] if len(thresholds) > 1 else axes[j]
            ax.imshow(img, cmap='gray')
            for x, y in corner_coords:
                ax.plot(x, y, 'ro', markersize=3)
            ax.set_title(f"k={k}, threshold={threshold}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("harris_test.jpg")
    plt.close()
    
    print("Harris test complete. Results saved to 'harris_test.jpg'")

def test_descriptors():
    """Test feature descriptors."""
    print("Testing feature descriptors...")
    
    # Load test image
    img_path = "data/image_pairs/test_image.jpg"
    if not os.path.exists(img_path):
        print(f"Test image not found at {img_path}. Using placeholder...")
        # Create a simple test image if none exists
        img = np.zeros((300, 300), dtype=np.uint8)
        img[100:200, 100:200] = 255  # White square in the middle
        cv2.imwrite(img_path, img)
    
    img = load_image(img_path)
    
    # Test Harris keypoint extraction
    harris_detector = HarrisDetector(k=0.04, threshold=0.01, window_size=3)
    keypoint_extractor = HarrisKeypointExtractor(harris_detector)
    keypoints = keypoint_extractor.detect(img)
    
    # Test SIFT descriptor
    sift_descriptor = FeatureDescriptor(descriptor_type='SIFT')
    keypoints_sift, descriptors_sift = sift_descriptor.compute_for_keypoints(img, keypoints)
    
    # Draw keypoints
    img_sift = cv2.drawKeypoints(img, keypoints_sift, None, 
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Try SURF descriptor if available
    try:
        surf_descriptor = FeatureDescriptor(descriptor_type='SURF')
        keypoints_surf, descriptors_surf = surf_descriptor.compute_for_keypoints(img, keypoints)
        
        # Draw keypoints
        img_surf = cv2.drawKeypoints(img, keypoints_surf, None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"SIFT Keypoints: {len(keypoints_sift)}")
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(img_surf, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"SURF Keypoints: {len(keypoints_surf)}")
        axes[1].axis('off')
    except:
        # If SURF is not available
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
        ax.set_title(f"SIFT Keypoints: {len(keypoints_sift)}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("descriptors_test.jpg")
    plt.close()
    
    print("Descriptors test complete. Results saved to 'descriptors_test.jpg'")

def test_matching():
    """Test feature matching and RANSAC."""
    print("Testing feature matching and RANSAC...")
    
    # Load two test images
    img1_path = "data/image_pairs/test_image1.jpg"
    img2_path = "data/image_pairs/test_image2.jpg"
    
    # Check if test images exist
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Test images not found. Using placeholders...")
        
        # Create simple test images if none exist
        img1 = np.zeros((300, 300), dtype=np.uint8)
        img1[100:200, 100:200] = 255  # White square in the middle
        
        # Second image with transformed square
        img2 = np.zeros((300, 300), dtype=np.uint8)
        img2[120:220, 80:180] = 255  # Shifted square
        
        cv2.imwrite(img1_path, img1)
        cv2.imwrite(img2_path, img2)
    
    # Load images
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    # Detect Harris corners
    harris_detector = HarrisDetector(k=0.04, threshold=0.01, window_size=3)
    keypoint_extractor = HarrisKeypointExtractor(harris_detector)
    
    keypoints1 = keypoint_extractor.detect(img1)
    keypoints2 = keypoint_extractor.detect(img2)
    
    # Compute SIFT descriptors
    descriptor = FeatureDescriptor(descriptor_type='SIFT')
    keypoints1, descriptors1 = descriptor.compute_for_keypoints(img1, keypoints1)
    keypoints2, descriptors2 = descriptor.compute_for_keypoints(img2, keypoints2)
    
    # Match features
    matcher = FeatureMatcher(ratio_threshold=0.75)
    matches = matcher.match_descriptors(descriptors1, descriptors2)
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
    
    # Apply RANSAC if enough matches
    if len(matches) >= 4:
        # Extract matched points
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Apply RANSAC
        ransac = RANSAC(n_iterations=1000, inlier_threshold=3.0)
        H, inliers = ransac.estimate_homography(points1, points2)
        
        # Draw matches with inliers
        if inliers is not None:
            mask = inliers.astype(int).tolist()
            img_ransac = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, 
                                        matchesMask=mask, flags=2)
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"All Matches: {len(matches)}")
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(img_ransac, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"RANSAC Inliers: {np.sum(inliers)}")
            axes[1].axis('off')
        else:
            # Create figure with only matches
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Matches: {len(matches)}, RANSAC failed")
            ax.axis('off')
    else:
        # Create figure with only matches
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Matches: {len(matches)}, not enough for RANSAC")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("matching_test.jpg")
    plt.close()
    
    print("Matching test complete. Results saved to 'matching_test.jpg'")

if __name__ == "__main__":
    # Ensure test directory exists
    Path("data/image_pairs").mkdir(parents=True, exist_ok=True)
    
    # Run tests
    test_harris()
    test_descriptors()
    test_matching()
    
    print("All tests complete!")