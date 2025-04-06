import numpy as np
import cv2

def load_image(image_path, grayscale=False):
    """
    Load an image from a file.
    
    Args:
        image_path (str): Path to the image file
        grayscale (bool): Whether to load the image in grayscale
    
    Returns:
        numpy.ndarray: Loaded image
    """
    if grayscale:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(image_path)

def load_homography(homography_path):
    """
    Load a homography matrix from a file.
    
    Args:
        homography_path (str): Path to the homography file
    
    Returns:
        numpy.ndarray: Loaded homography matrix
    """
    return np.loadtxt(homography_path)

def resize_image(image, max_size=1000):
    """
    Resize the image while preserving aspect ratio.
    
    Args:
        image (numpy.ndarray): Input image
        max_size (int): Maximum dimension (width or height)
    
    Returns:
        numpy.ndarray: Resized image
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))
    
    return cv2.resize(image, (new_width, new_height))

def extract_matched_points(keypoints1, keypoints2, matches):
    """
    Extract matched points from keypoints and matches.
    
    Args:
        keypoints1 (list): List of keypoints in the first image
        keypoints2 (list): List of keypoints in the second image
        matches (list): List of DMatch objects
    
    Returns:
        tuple: (points1, points2) arrays of matched points
    """
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])
    
    return points1, points2

def save_keypoints(keypoints, filename):
    """
    Save keypoints to a file.
    
    Args:
        keypoints (list): List of cv2.KeyPoint objects
        filename (str): Path to save the keypoints
    """
    # Convert keypoints to a format that can be saved
    keypoints_data = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) 
                      for kp in keypoints]
    np.save(filename, keypoints_data)

def load_keypoints(filename):
    """
    Load keypoints from a file.
    
    Args:
        filename (str): Path to the keypoints file
    
    Returns:
        list: List of cv2.KeyPoint objects
    """
    keypoints_data = np.load(filename)
    keypoints = [cv2.KeyPoint(x=float(x), y=float(y), _size=float(size), 
                              _angle=float(angle), _response=float(response), 
                              _octave=int(octave), _class_id=int(class_id)) 
                for x, y, size, angle, response, octave, class_id in keypoints_data]
    return keypoints