import numpy as np
import cv2
from scipy.spatial.distance import cdist

class FeatureMatcher:
    def __init__(self, ratio_threshold=0.75, distance_metric='euclidean'):
        """
        Initialize feature matcher.
        
        Args:
            ratio_threshold (float): Threshold for Lowe's ratio test
            distance_metric (str): Distance metric for descriptor matching
        """
        self.ratio_threshold = ratio_threshold
        self.distance_metric = distance_metric
    
    def match_descriptors(self, desc1, desc2):
        """
        Match descriptors using Lowe's ratio test.
        
        Args:
            desc1 (numpy.ndarray): First set of descriptors
            desc2 (numpy.ndarray): Second set of descriptors
            
        Returns:
            list: List of DMatch objects
        """
        # Compute distance matrix
        distances = cdist(desc1, desc2, metric=self.distance_metric)
        
        # Find matches using ratio test
        matches = []
        for i in range(distances.shape[0]):
            # Get sorted indices of distances for current descriptor
            sorted_indices = np.argsort(distances[i])
            
            # Check if we have at least 2 matches for ratio test
            if len(sorted_indices) < 2:
                continue
            
            # Get best and second best match distances
            best_match_idx = sorted_indices[0]
            second_best_match_idx = sorted_indices[1]
            
            best_distance = distances[i, best_match_idx]
            second_best_distance = distances[i, second_best_match_idx]
            
            # Apply ratio test
            if best_distance < self.ratio_threshold * second_best_distance:
                matches.append(cv2.DMatch(
                    _queryIdx=i,
                    _trainIdx=best_match_idx,
                    _distance=best_distance
                ))
        
        return matches

class RANSAC:
    def __init__(self, n_iterations=1000, inlier_threshold=3.0, min_inliers=10):
        """
        Initialize RANSAC algorithm for homography estimation.
        
        Args:
            n_iterations (int): Number of RANSAC iterations
            inlier_threshold (float): Threshold for inlier identification
            min_inliers (int): Minimum number of inliers for a valid model
        """
        self.n_iterations = n_iterations
        self.inlier_threshold = inlier_threshold
        self.min_inliers = min_inliers
    
    def estimate_homography(self, src_points, dst_points):
        """
        Estimate homography matrix using RANSAC.
        
        Args:
            src_points (numpy.ndarray): Source points (N, 2)
            dst_points (numpy.ndarray): Destination points (N, 2)
            
        Returns:
            tuple: (H, inliers) where H is the homography matrix and
                  inliers is a binary mask of inlier matches
        """
        assert src_points.shape[0] == dst_points.shape[0], "Number of points must match"
        assert src_points.shape[0] >= 4, "At least 4 point pairs are required"
        
        n_points = src_points.shape[0]
        
        # Convert points to homogeneous coordinates
        src_homogeneous = np.hstack((src_points, np.ones((n_points, 1))))
        
        best_H = None
        best_inliers = None
        best_inlier_count = 0
        
        for _ in range(self.n_iterations):
            # Randomly select 4 point pairs
            indices = np.random.choice(n_points, 4, replace=False)
            sampled_src = src_points[indices]
            sampled_dst = dst_points[indices]
            
            # Calculate homography for the sampled points
            try:
                H, _ = cv2.findHomography(sampled_src, sampled_dst, 0)
                
                if H is None:
                    continue
                
                # Transform source points
                transformed_src = cv2.perspectiveTransform(
                    src_points.reshape(-1, 1, 2), H
                ).reshape(-1, 2)
                
                # Calculate distances
                distances = np.sqrt(np.sum((transformed_src - dst_points) ** 2, axis=1))
                
                # Identify inliers
                inliers = distances < self.inlier_threshold
                inlier_count = np.sum(inliers)
                
                # Update best model if necessary
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_inliers = inliers
                    best_H = H
            
            except cv2.error:
                # Skip if homography estimation fails
                continue
        
        # Refine homography using all inliers if we have a good model
        if best_inlier_count >= self.min_inliers:
            src_inliers = src_points[best_inliers]
            dst_inliers = dst_points[best_inliers]
            
            refined_H, _ = cv2.findHomography(src_inliers, dst_inliers, 0)
            
            if refined_H is not None:
                best_H = refined_H
        
        return best_H, best_inliers
    
    def compute_match_quality(self, H, src_points, dst_points, inliers):
        """
        Compute match quality score based on homography transformation.
        
        Args:
            H (numpy.ndarray): Homography matrix
            src_points (numpy.ndarray): Source points
            dst_points (numpy.ndarray): Destination points
            inliers (numpy.ndarray): Binary mask of inlier matches
            
        Returns:
            float: Match quality score
        """
        if H is None or np.sum(inliers) < self.min_inliers:
            return 0.0
        
        # Transform all source points using homography
        transformed_src = cv2.perspectiveTransform(
            src_points.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        
        # Calculate distances for inliers
        inlier_distances = np.sqrt(np.sum(
            (transformed_src[inliers] - dst_points[inliers]) ** 2, axis=1
        ))
        
        # Calculate quality metrics
        inlier_ratio = np.mean(inliers)
        mean_inlier_distance = np.mean(inlier_distances)
        
        # Compute quality score (higher is better)
        quality_score = inlier_ratio * (1.0 / (1.0 + mean_inlier_distance))
        
        return quality_score
    

# import numpy as np
# import cv2
# from scipy.spatial.distance import cdist

# class FeatureMatcher:
#     def __init__(self, ratio_threshold=0.75, distance_metric='euclidean'):
#         """
#         Initialize feature matcher.
        
#         Args:
#             ratio_threshold (float): Threshold for Lowe's ratio test
#             distance_metric (str): Distance metric for descriptor matching
#         """
#         self.ratio_threshold = ratio_threshold
#         self.distance_metric = distance_metric
    
#     def match_descriptors(self, desc1, desc2):
#         """
#         Match descriptors using Lowe's ratio test.
        
#         Args:
#             desc1 (numpy.ndarray): First set of descriptors
#             desc2 (numpy.ndarray): Second set of descriptors
            
#         Returns:
#             list: List of DMatch objects
#         """
#         # TODO: Implement descriptor matching with Lowe's ratio test
#         # HINT: Compute distances between all descriptor pairs and apply ratio test
        
#         # Compute distance matrix
#         # Your implementation here
#         distances = None
        
#         # Find matches using ratio test
#         matches = []
        
#         # Your implementation of ratio test here
        
#         return matches

# class RANSAC:
#     def __init__(self, n_iterations=1000, inlier_threshold=3.0, min_inliers=10):
#         """
#         Initialize RANSAC algorithm for homography estimation.
        
#         Args:
#             n_iterations (int): Number of RANSAC iterations
#             inlier_threshold (float): Threshold for inlier identification
#             min_inliers (int): Minimum number of inliers for a valid model
#         """
#         self.n_iterations = n_iterations
#         self.inlier_threshold = inlier_threshold
#         self.min_inliers = min_inliers
    
#     def estimate_homography(self, src_points, dst_points):
#         """
#         Estimate homography matrix using RANSAC.
        
#         Args:
#             src_points (numpy.ndarray): Source points (N, 2)
#             dst_points (numpy.ndarray): Destination points (N, 2)
            
#         Returns:
#             tuple: (H, inliers) where H is the homography matrix and
#                   inliers is a binary mask of inlier matches
#         """
#         assert src_points.shape[0] == dst_points.shape[0], "Number of points must match"
#         assert src_points.shape[0] >= 4, "At least 4 point pairs are required"
        
#         # TODO: Implement RANSAC algorithm for homography estimation
#         # HINT: 1. Randomly select 4 point pairs
#         #       2. Compute homography
#         #       3. Transform all points
#         #       4. Identify inliers
#         #       5. Keep the best model
        
#         n_points = src_points.shape[0]
        
#         # Your implementation here
#         best_H = None
#         best_inliers = None
        
#         return best_H, best_inliers
    
#     def compute_match_quality(self, H, src_points, dst_points, inliers):
#         """
#         Compute match quality score based on homography transformation.
        
#         Args:
#             H (numpy.ndarray): Homography matrix
#             src_points (numpy.ndarray): Source points
#             dst_points (numpy.ndarray): Destination points
#             inliers (numpy.ndarray): Binary mask of inlier matches
            
#         Returns:
#             float: Match quality score
#         """
#         # TODO: Implement match quality evaluation
#         # HINT: Consider inlier ratio and transformation error
        
#         # Your implementation here
#         quality_score = 0.0
        
#         return quality_score