import numpy as np
import cv2

class HarrisDetector:
    def __init__(self, k=0.04, window_size=3, threshold=0.01):
        """
        Initialize Harris corner detector.
        
        Args:
            k (float): Harris detector free parameter, typically in range [0.04, 0.06]
            window_size (int): Window size for computing the response
            threshold (float): Threshold for corner detection
        """
        self.k = k
        self.window_size = window_size
        self.threshold = threshold
        
    def compute_gradients(self, image):
        """
        Compute image gradients using Sobel operators.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            tuple: (dx, dy) gradient images
        """
        # Implement gradient computation using Sobel operators
        # HINT: Use cv2.Sobel() for gradient computation
        
        # Convert to float for better precision
        image = image.astype(np.float32)
        
        # Your implementation here
        dx = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0)
        dy = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1)
        return dx, dy
    
    def compute_structure_tensor(self, dx, dy):
        """
        Compute structure tensor for Harris corner detection.
        
        Args:
            dx (numpy.ndarray): X-gradient image
            dy (numpy.ndarray): Y-gradient image
            
        Returns:
            tuple: (Ixx, Ixy, Iyy) structure tensor components
        """
        # Implement structure tensor computation
        # HINT: Apply Gaussian smoothing to the products of derivatives
    
        # Apply Gaussian smoothing
        # Your implementation here
        kernel_size = (3,3)
        sigma = 1

        Ixx = cv2.GaussianBlur(dx**2, kernel_size, sigma)
        Ixy = cv2.GaussianBlur(dx*dy, kernel_size, sigma)
        Iyy = cv2.GaussianBlur(dy**2, kernel_size, sigma)
        
        return Ixx, Ixy, Iyy
    
    def compute_corner_response(self, Ixx, Ixy, Iyy, k=0.04):
        """
        Compute Harris corner response.
        
        Args:
            Ixx (numpy.ndarray): Structure tensor component
            Ixy (numpy.ndarray): Structure tensor component
            Iyy (numpy.ndarray): Structure tensor component
            
        Returns:
            numpy.ndarray: Corner response image
        """
        # Implement Harris corner response computation
        # HINT: R = det(M) - k * trace(M)^2
        # where M = [[Ixx, Ixy], [Ixy, Iyy]]

        # Your implementation here
        det_M = Ixx * Iyy - Ixy ** 2 
        trace_M = Ixx + Iyy
        R = det_M - k * (trace_M ** 2)
        
        return R
    
    def non_max_suppression(self, response, neighborhood_size=3):
        """
        Apply non-maximum suppression to corner response.
        
        Args:
            response (numpy.ndarray): Corner response image
            neighborhood_size (int): Size of the neighborhood for suppression
            
        Returns:
            numpy.ndarray: Binary image with corners after suppression
        """
        # Implement non-maximum suppression
        # HINT: For each pixel, check if it's the maximum in its neighborhood
        
        # Normalize response to [0, 1]
        if response.max() > 0:
            response_normalized = response / response.max()
        else:
            response_normalized = response
        
        # Apply threshold
        corners = response_normalized > self.threshold
    
        # Your implementation of non-maximum suppression here
        # result = np.zeros_like(corners, dtype=bool)
        dilated_response = cv2.dilate(response_normalized, None)
        result = (response_normalized == dilated_response) & (response_normalized > self.threshold)
        n_halfsize = neighborhood_size // 2
        for i in range(n_halfsize, response.shape[0] - n_halfsize):
            for j in range(n_halfsize, response.shape[1] - n_halfsize):
                neighborhood = response_normalized[i - n_halfsize:i + n_halfsize + 1,
                                                   j - n_halfsize:j + n_halfsize + 1]
                # check if current pixel is maximum
                if response_normalized[i, j] == np.max(neighborhood):
                    result[i, j] = corners[i, j]
        
        return result
    
    def detect_corners(self, image):
        """
        Detect corners in the input image.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            tuple: (corners, response) where corners is a binary image with detected 
                  corners and response is the Harris response image
        """
        # Ensure the image is grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients
        dx, dy = self.compute_gradients(image)
        
        # Compute structure tensor
        Ixx, Ixy, Iyy = self.compute_structure_tensor(dx, dy)
        
        # Compute corner response
        response = self.compute_corner_response(Ixx, Ixy, Iyy)
        
        # Apply non-maximum suppression
        corners = self.non_max_suppression(response)
        
        return corners, response
    
    def get_corner_coordinates(self, corners):
        """
        Convert binary corner image to list of coordinates.
        
        Args:
            corners (numpy.ndarray): Binary image with corners
            
        Returns:
            list: List of (x, y) corner coordinates
        """
        y_coords, x_coords = np.where(corners)
        return [(x, y) for x, y in zip(x_coords, y_coords)]