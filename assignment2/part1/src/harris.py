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
        # Convert to float for better precision
        image = image.astype(np.float32)
        
        # Compute gradients using Sobel operators
        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
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
        # Compute products of derivatives
        Ixx = dx * dx
        Ixy = dx * dy
        Iyy = dy * dy
        
        # Apply Gaussian smoothing
        Ixx = cv2.GaussianBlur(Ixx, (self.window_size, self.window_size), 0)
        Ixy = cv2.GaussianBlur(Ixy, (self.window_size, self.window_size), 0)
        Iyy = cv2.GaussianBlur(Iyy, (self.window_size, self.window_size), 0)
        
        return Ixx, Ixy, Iyy
    
    def compute_corner_response(self, Ixx, Ixy, Iyy):
        """
        Compute Harris corner response.
        
        Args:
            Ixx (numpy.ndarray): Structure tensor component
            Ixy (numpy.ndarray): Structure tensor component
            Iyy (numpy.ndarray): Structure tensor component
            
        Returns:
            numpy.ndarray: Corner response image
        """
        # Compute determinant and trace
        det = Ixx * Iyy - Ixy ** 2
        trace = Ixx + Iyy
        
        # Compute Harris response
        response = det - self.k * (trace ** 2)
        
        return response
    
    def non_max_suppression(self, response, neighborhood_size=3):
        """
        Apply non-maximum suppression to corner response.
        
        Args:
            response (numpy.ndarray): Corner response image
            neighborhood_size (int): Size of the neighborhood for suppression
            
        Returns:
            numpy.ndarray: Binary image with corners after suppression
        """
        # Normalize response to [0, 1]
        if response.max() > 0:
            response_normalized = response / response.max()
        else:
            response_normalized = response
        
        # Apply threshold
        corners = response_normalized > self.threshold
        
        # Non-maximum suppression
        height, width = response.shape
        result = np.zeros_like(corners, dtype=bool)
        
        for y in range(height):
            for x in range(width):
                if corners[y, x]:
                    # Define neighborhood boundaries
                    y_min = max(0, y - neighborhood_size // 2)
                    y_max = min(height, y + neighborhood_size // 2 + 1)
                    x_min = max(0, x - neighborhood_size // 2)
                    x_max = min(width, x + neighborhood_size // 2 + 1)
                    
                    # Extract neighborhood
                    neighborhood = response_normalized[y_min:y_max, x_min:x_max]
                    
                    # Check if current point is maximum in neighborhood
                    if response_normalized[y, x] == neighborhood.max():
                        result[y, x] = True
        
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
    

# import numpy as np
# import cv2

# class HarrisDetector:
#     def __init__(self, k=0.04, window_size=3, threshold=0.01):
#         """
#         Initialize Harris corner detector.
        
#         Args:
#             k (float): Harris detector free parameter, typically in range [0.04, 0.06]
#             window_size (int): Window size for computing the response
#             threshold (float): Threshold for corner detection
#         """
#         self.k = k
#         self.window_size = window_size
#         self.threshold = threshold
        
#     def compute_gradients(self, image):
#         """
#         Compute image gradients using Sobel operators.
        
#         Args:
#             image (numpy.ndarray): Input grayscale image
            
#         Returns:
#             tuple: (dx, dy) gradient images
#         """
#         # TODO: Implement gradient computation using Sobel operators
#         # HINT: Use cv2.Sobel() for gradient computation
        
#         # Convert to float for better precision
#         image = image.astype(np.float32)
        
#         # Your implementation here
#         dx = None
#         dy = None
        
#         return dx, dy
    
#     def compute_structure_tensor(self, dx, dy):
#         """
#         Compute structure tensor for Harris corner detection.
        
#         Args:
#             dx (numpy.ndarray): X-gradient image
#             dy (numpy.ndarray): Y-gradient image
            
#         Returns:
#             tuple: (Ixx, Ixy, Iyy) structure tensor components
#         """
#         # TODO: Implement structure tensor computation
#         # HINT: Apply Gaussian smoothing to the products of derivatives
        
#         # Compute products of derivatives
#         Ixx = None
#         Ixy = None
#         Iyy = None
        
#         # Apply Gaussian smoothing
#         # Your implementation here
        
#         return Ixx, Ixy, Iyy
    
#     def compute_corner_response(self, Ixx, Ixy, Iyy):
#         """
#         Compute Harris corner response.
        
#         Args:
#             Ixx (numpy.ndarray): Structure tensor component
#             Ixy (numpy.ndarray): Structure tensor component
#             Iyy (numpy.ndarray): Structure tensor component
            
#         Returns:
#             numpy.ndarray: Corner response image
#         """
#         # TODO: Implement Harris corner response computation
#         # HINT: R = det(M) - k * trace(M)^2
#         # where M = [[Ixx, Ixy], [Ixy, Iyy]]
        
#         # Your implementation here
#         response = None
        
#         return response
    
#     def non_max_suppression(self, response, neighborhood_size=3):
#         """
#         Apply non-maximum suppression to corner response.
        
#         Args:
#             response (numpy.ndarray): Corner response image
#             neighborhood_size (int): Size of the neighborhood for suppression
            
#         Returns:
#             numpy.ndarray: Binary image with corners after suppression
#         """
#         # TODO: Implement non-maximum suppression
#         # HINT: For each pixel, check if it's the maximum in its neighborhood
        
#         # Normalize response to [0, 1]
#         if response.max() > 0:
#             response_normalized = response / response.max()
#         else:
#             response_normalized = response
        
#         # Apply threshold
#         corners = response_normalized > self.threshold
        
#         # Your implementation of non-maximum suppression here
#         result = np.zeros_like(corners, dtype=bool)
        
#         return result
    
#     def detect_corners(self, image):
#         """
#         Detect corners in the input image.
        
#         Args:
#             image (numpy.ndarray): Input grayscale image
            
#         Returns:
#             tuple: (corners, response) where corners is a binary image with detected 
#                   corners and response is the Harris response image
#         """
#         # Ensure the image is grayscale
#         if len(image.shape) > 2:
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Compute gradients
#         dx, dy = self.compute_gradients(image)
        
#         # Compute structure tensor
#         Ixx, Ixy, Iyy = self.compute_structure_tensor(dx, dy)
        
#         # Compute corner response
#         response = self.compute_corner_response(Ixx, Ixy, Iyy)
        
#         # Apply non-maximum suppression
#         corners = self.non_max_suppression(response)
        
#         return corners, response
    
#     def get_corner_coordinates(self, corners):
#         """
#         Convert binary corner image to list of coordinates.
        
#         Args:
#             corners (numpy.ndarray): Binary image with corners
            
#         Returns:
#             list: List of (x, y) corner coordinates
#         """
#         y_coords, x_coords = np.where(corners)
#         return [(x, y) for x, y in zip(x_coords, y_coords)]