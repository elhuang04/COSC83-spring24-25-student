import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Import your implementations
from src.harris import HarrisDetector
from src.descriptors import FeatureDescriptor, HarrisKeypointExtractor
from src.matching import FeatureMatcher, RANSAC
from src.visualization import (
    visualize_corners, visualize_keypoints, visualize_matches,
    visualize_harris_response, create_match_ranking_visualization
)
from utils.image_utils import (
    load_image, resize_image, extract_matched_points
)

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def process_image_pair(img1_path, img2_path, harris_detector, descriptor_type, results_dir):
    """
    Process a pair of images through the feature detection, description and matching pipeline.
    
    Args:
        img1_path (str): Path to the first image
        img2_path (str): Path to the second image
        harris_detector (HarrisDetector): Harris detector instance
        descriptor_type (str): Type of descriptor to use ('SIFT' or 'SURF')
        results_dir (str): Directory to save results
        
    Returns:
        dict: Dictionary with results and metrics
    """
    # Load and resize images
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    img1 = resize_image(img1, max_size=800)
    img2 = resize_image(img2, max_size=800)
    
    # Create result directories
    pair_name = f"{os.path.basename(img1_path).split('.')[0]}_{os.path.basename(img2_path).split('.')[0]}"
    pair_dir = os.path.join(results_dir, pair_name)
    ensure_dir(pair_dir)
    
    # Step 1: Harris Corner Detection
    print(f"Processing pair: {pair_name}")
    print("Step 1: Harris Corner Detection")
    corners1, response1 = harris_detector.detect_corners(img1)
    corners2, response2 = harris_detector.detect_corners(img2)

    print("CORNERS",corners1, "RESPONSE", response1)
    
    # Visualize Harris corners
    corners_vis1 = visualize_corners(img1, corners1)
    corners_vis2 = visualize_corners(img2, corners2)
    
    cv2.imwrite(os.path.join(pair_dir, "harris_corners1.jpg"), corners_vis1)
    cv2.imwrite(os.path.join(pair_dir, "harris_corners2.jpg"), corners_vis2)
    
    # Visualize Harris response
    response_vis1 = visualize_harris_response(response1)
    response_vis2 = visualize_harris_response(response2)
    
    cv2.imwrite(os.path.join(pair_dir, "harris_response1.jpg"), response_vis1)
    cv2.imwrite(os.path.join(pair_dir, "harris_response2.jpg"), response_vis2)
    
    # Step 2: Feature Description
    print("Step 2: Feature Description")
    
    # Convert Harris corners to keypoints
    keypoint_extractor = HarrisKeypointExtractor(harris_detector)
    keypoints1 = keypoint_extractor.detect(img1)
    keypoints2 = keypoint_extractor.detect(img2)

    
    # print(keypoints1)
    
    # Compute descriptors
    descriptor = FeatureDescriptor(descriptor_type=descriptor_type)
    keypoints1, descriptors1 = descriptor.compute_for_keypoints(img1, keypoints1)
    keypoints2, descriptors2 = descriptor.compute_for_keypoints(img2, keypoints2)

    print("DESCRIPTOR", descriptor)
    print("DESCRIPTORS 1", descriptors1)
    
    # Visualize keypoints
    keypoints_vis1 = visualize_keypoints(img1, keypoints1)
    keypoints_vis2 = visualize_keypoints(img2, keypoints2)
    
    cv2.imwrite(os.path.join(pair_dir, f"{descriptor_type.lower()}_keypoints1.jpg"), keypoints_vis1)
    cv2.imwrite(os.path.join(pair_dir, f"{descriptor_type.lower()}_keypoints2.jpg"), keypoints_vis2)
    
    # Step 3: Feature Matching with RANSAC
    print("Step 3: Feature Matching with RANSAC")
    
    # Match descriptors
    matcher = FeatureMatcher(ratio_threshold=0.75)
    matches = matcher.match_descriptors(descriptors1, descriptors2)
    
    print("MATCHER", matcher)
    print("MATCHER", matches)
    
    # Visualize initial matches
    initial_matches_vis = visualize_matches(img1, keypoints1, img2, keypoints2, matches)
    cv2.imwrite(os.path.join(pair_dir, "initial_matches.jpg"), initial_matches_vis)
    
    # Extract matched points
    if len(matches) >= 4:
        points1, points2 = extract_matched_points(keypoints1, keypoints2, matches)
        
        # Apply RANSAC
        ransac = RANSAC(n_iterations=1000, inlier_threshold=3.0)
        H, inliers = ransac.estimate_homography(points1, points2)
        
        # Visualize ransac matches
        ransac_matches_vis = visualize_matches(img1, keypoints1, img2, keypoints2, matches, inliers)
        cv2.imwrite(os.path.join(pair_dir, "ransac_matches.jpg"), ransac_matches_vis)
        
        # Compute match quality
        quality_score = ransac.compute_match_quality(H, points1, points2, inliers)
        
        inlier_count = np.sum(inliers) if inliers is not None else 0
        inlier_ratio = inlier_count / len(matches) if len(matches) > 0 else 0
        
        print(f"Matches: {len(matches)}, Inliers: {inlier_count}, Ratio: {inlier_ratio:.2f}, Quality: {quality_score:.4f}")
        
        # Return results
        return {
            'pair_name': pair_name,
            'corners1': corners1,
            'corners2': corners2,
            'keypoints1': keypoints1,
            'keypoints2': keypoints2,
            'matches': matches,
            'inliers': inliers,
            'homography': H,
            'match_count': len(matches),
            'inlier_count': inlier_count,
            'inlier_ratio': inlier_ratio,
            'quality_score': quality_score,
            'matches_vis': ransac_matches_vis
        }
    else:
        print(f"Not enough matches for RANSAC: {len(matches)}")
        return {
            'pair_name': pair_name,
            'corners1': corners1,
            'corners2': corners2,
            'keypoints1': keypoints1,
            'keypoints2': keypoints2,
            'matches': matches,
            'inliers': None,
            'homography': None,
            'match_count': len(matches),
            'inlier_count': 0,
            'inlier_ratio': 0,
            'quality_score': 0,
            'matches_vis': initial_matches_vis
        }


def main():
    # Create directories
    results_dir = "results"
    ensure_dir(results_dir)
    
    # Directory for image pairs
    data_dir = "data/image_pairs"
    
    # Get image pairs
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Found {len(image_files)} images in {data_dir}")
    # Create pairs (for this example, we're assuming consecutive images form a pair)
    # In a real assignment, you might have a specific pairing scheme
    image_pairs = []
    for i in range(0, len(image_files) - 1, 2):
        if i + 1 < len(image_files):
            image_pairs.append((
                os.path.join(data_dir, image_files[i]),
                os.path.join(data_dir, image_files[i + 1])
            ))
    
    # Initialize Harris detector
    harris_detector = HarrisDetector(k=0.04, window_size=3, threshold=0.01)
    
    # Choose descriptor type ('SIFT' or 'SURF')
    descriptor_type = 'SIFT'
    
    # Process each pair
    pair_results = []
    for img1_path, img2_path in image_pairs:
        result = process_image_pair(img1_path, img2_path, harris_detector, descriptor_type, results_dir)
        pair_results.append(result)
    
    # Create ranking visualization
    if pair_results:
        # Extract quality scores and matching visualizations
        quality_scores = [result['quality_score'] for result in pair_results]
        matches_vis = [(None, None, result['matches_vis']) for result in pair_results]
        
        # Create ranking visualization
        ranking_vis = create_match_ranking_visualization(matches_vis, quality_scores)
        cv2.imwrite(os.path.join(results_dir, "ranking_visualization.jpg"), ranking_vis)
        
        # Save ranking data
        ranking_data = []
        for result in pair_results:
            ranking_data.append({
                'pair_name': result['pair_name'],
                'match_count': result['match_count'],
                'inlier_count': result['inlier_count'],
                'inlier_ratio': result['inlier_ratio'],
                'quality_score': result['quality_score']
            })
        
        # Sort by quality score
        ranking_data = sorted(ranking_data, key=lambda x: x['quality_score'], reverse=True)
        
        # Print ranking
        print("\n--- Image Pair Ranking ---")
        for i, data in enumerate(ranking_data):
            print(f"{i+1}. {data['pair_name']}: Quality={data['quality_score']:.4f}, "
                  f"Inliers={data['inlier_count']}/{data['match_count']}, "
                  f"Ratio={data['inlier_ratio']:.2f}")


if __name__ == "__main__":
    main()