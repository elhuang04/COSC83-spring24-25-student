import numpy as np
import cv2

class FeatureDescriptor:
    def __init__(self, descriptor_type='SIFT', params=None):
        """
        Initialize feature descriptor.
        
        Args:
            descriptor_type (str): Type of descriptor ('SIFT' or 'SURF')
            params (dict): Parameters for the descriptor
        """
        self.descriptor_type = descriptor_type
        self.params = params if params is not None else {}
        
        self._init_descriptor()
    
    def _init_descriptor(self):
        """
        Initialize the descriptor object based on the type.
        """
        # TODO: Initialize SIFT or SURF descriptor based on self.descriptor_type
        # HINT: Use cv2.SIFT_create() or cv2.xfeatures2d.SURF_create()
        
        if self.descriptor_type == 'SIFT':
            # Extract parameters from self.params or use default values
            
            # Your implementation here
            self.descriptor = None
            
        elif self.descriptor_type == 'SURF':
            # Extract parameters from self.params or use default values
            
            # Your implementation here
            self.descriptor = None
            
        else:
            raise ValueError(f"Unsupported descriptor type: {self.descriptor_type}")
    
    def detect_and_compute(self, image, mask=None):
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            mask (numpy.ndarray): Optional mask to restrict feature detection
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        # Ensure the image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # TODO: Use the descriptor to detect keypoints and compute descriptors
        # HINT: Use the detectAndCompute method
        
        # Your implementation here
        keypoints, descriptors = None, None
        
        return keypoints, descriptors
    
    def compute_for_keypoints(self, image, keypoints):
        """
        Compute descriptors for given keypoints.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            keypoints (list): List of keypoints
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        # Ensure the image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # TODO: Compute descriptors for the provided keypoints
        # HINT: Use the compute method
        
        # Your implementation here
        keypoints, descriptors = None, None
        
        return keypoints, descriptors

class HarrisKeypointExtractor:
    def __init__(self, harris_detector):
        """
        Initialize keypoint extractor based on Harris detector.
        
        Args:
            harris_detector (HarrisDetector): Harris corner detector instance
        """
        self.harris_detector = harris_detector
    
    def detect(self, image, mask=None):
        """
        Detect keypoints using Harris detector.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            mask (numpy.ndarray): Optional mask to restrict feature detection
            
        Returns:
            list: List of cv2.KeyPoint objects
        """
        # TODO: Detect Harris corners and convert them to cv2.KeyPoint objects
        # HINT: Use the harris_detector to find corners, then convert coordinates to KeyPoints
        
        # Your implementation here
        keypoints = []
        
        return keypoints