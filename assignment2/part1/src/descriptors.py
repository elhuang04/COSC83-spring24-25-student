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
        # Initialize SIFT or SURF descriptor based on self.descriptor_type
        # HINT: Use cv2.SIFT_create() or cv2.xfeatures2d.SURF_create()
        
        if self.descriptor_type == 'SIFT':
            # Extract parameters from self.params or use default values
            nfeatures = self.params.get('nfeatures', 100)
            nOctaveLayers = self.params.get('nOctaveLayers', 3)
            contrastThreshold = self.params.get('contrastThreshold', 0.04)
            edgeThreshold = self.params.get('edgeThreshold', 10)
            sigma = self.params.get('sigma', 1.6)

            # Initialize the SIFT descriptor
            self.descriptor = cv2.SIFT_create(
                nfeatures=nfeatures,
                nOctaveLayers=nOctaveLayers,
                contrastThreshold=contrastThreshold,
                edgeThreshold=edgeThreshold,
                sigma=sigma
            )
            
        elif self.descriptor_type == 'SURF':
            # Extract parameters from self.params or use default values
            hessianThreshold = self.params.get('hessianThreshold', 100)
            nOctaves = self.params.get('nOctaves', 4)
            nOctaveLayers = self.params.get('nOctaveLayers', 3)
            extended = self.params.get('extended', False)
            upright = self.params.get('upright', False)

            # Initialize the SURF descriptor
            self.descriptor = cv2.xfeatures2d.SURF_create(
                hessianThreshold=hessianThreshold,
                nOctaves=nOctaves,
                nOctaveLayers=nOctaveLayers,
                extended=extended,
                upright=upright
            )

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
        
        # Use the descriptor to detect keypoints and compute descriptors
        # HINT: Use the detectAndCompute method
        
        # Your implementation here
        keypoints, descriptors = self.descriptor.detectAndCompute(image, mask)

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
        
        # Compute descriptors for the provided keypoints
        # HINT: Use the compute method
        
        # Your implementation here
        keypoints, descriptors = self.descriptor.compute(image, keypoints)
        
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
        # Detect Harris corners and convert them to cv2.KeyPoint objects
        # HINT: Use the harris_detector to find corners, then convert coordinates to KeyPoints
        
        # Your implementation here
        corners, response = self.harris_detector.detect_corners(image)
        # harris_response = cv2.dilate(response, None)
        harris_response = response
        threshold = 0.01 * harris_response.max()

        keypoints = []
        for y in range(harris_response.shape[0]):
            for x in range(harris_response.shape[1]):
                if harris_response[y, x] > threshold:
                    if mask is None or mask[y, x]:
                        keypoints.append(cv2.KeyPoint(x=float(x), y=float(y), size=3.0))

        return keypoints