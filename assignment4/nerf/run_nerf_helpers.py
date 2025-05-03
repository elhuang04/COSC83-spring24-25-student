import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    """
    The Embedder class implements positional encoding from the NeRF paper.
    
    This class creates a function that transforms input coordinates to a higher dimensional space,
    which helps the network represent high-frequency functions more effectively.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Embedder with the provided parameters.
        
        Args:
            **kwargs: Dictionary containing configuration parameters:
                - input_dims: Dimensionality of input coordinates
                - include_input: Whether to include the original coordinates
                - max_freq_log2: Log2 of maximum frequency
                - num_freqs: Number of frequency bands
                - log_sampling: Whether to sample frequencies in log space
                - periodic_fns: List of periodic functions to use (sin, cos)
        """
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        """
        Create the embedding function based on the configuration.
        
        This function builds a list of embedding functions that will be applied to
        the input coordinates, and calculates the output dimensionality.
        """
        # TODO: Implement the create_embedding_fn method
        # 1. Initialize empty list to store embedding functions
        # 2. Get input dimensionality from kwargs
        # 3. Initialize output dimensionality counter
        # 4. If include_input is true, add the identity function to embedding functions
        # 5. Get max_freq_log2 and num_freqs from kwargs
        # 6. Create frequency bands based on log_sampling parameter
        # 7. For each frequency and periodic function, create and add embedding function
        # 8. Store embedding functions and output dimensionality in self
        
        pass  # Replace with your implementation
        
    def embed(self, inputs):
        """
        Apply the embedding functions to the inputs.
        
        Args:
            inputs: Input coordinates [batch_size, input_dims]
            
        Returns:
            Embedded coordinates [batch_size, out_dim]
        """
        # TODO: Implement the embed method
        # 1. Apply each embedding function to the inputs
        # 2. Concatenate the results along the last dimension
        # 3. Return the concatenated tensor
        
        pass  # Replace with your implementation


def get_embedder(multires, i=0):
    """
    Create an embedding function based on the specified parameters.
    
    Args:
        multires: Number of frequency bands (log2 of max frequency)
        i: Special parameter (-1 means identity embedding, 0 means normal embedding)
        
    Returns:
        embedding_function: Function that performs positional encoding
        out_dim: Output dimensionality of the encoding
    """
    # TODO: Implement the get_embedder function
    # 1. If i is -1, return identity function and input dimensionality (3)
    # 2. Create embedding kwargs dictionary with appropriate parameters
    # 3. Create Embedder object with kwargs
    # 4. Create and return lambda function that calls embedder.embed
    
    pass  # Replace with your implementation


# Model
class NeRF(nn.Module):
    """
    Neural Radiance Field (NeRF) model.
    
    This model predicts the color and density at each 3D point based on the position
    and viewing direction using a multi-layer perceptron (MLP).
    """
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        Initialize the NeRF model.
        
        Args:
            D: Network depth (number of layers)
            W: Network width (channels per layer)
            input_ch: Number of input channels for spatial coordinates
            input_ch_views: Number of input channels for viewing direction
            output_ch: Number of output channels (RGB + density)
            skips: List of layers with skip connections
            use_viewdirs: Whether to use viewing directions as input
        """
        super(NeRF, self).__init__()
        
        # TODO: Implement the NeRF model initialization
        # 1. Store input parameters as instance variables
        # 2. Create ModuleList for the position MLP (pts_linears)
        #    - First layer: input_ch → W
        #    - Middle layers: W → W (with skip connections at specified layers)
        # 3. Create ModuleList for the view direction MLP (views_linears)
        # 4. Create output layers based on whether viewdirs are used:
        #    - If use_viewdirs=True: 
        #        * feature_linear: W → W
        #        * alpha_linear: W → 1
        #        * rgb_linear: W//2 → 3
        #    - If use_viewdirs=False:
        #        * output_linear: W → output_ch
        
        pass  # Replace with your implementation

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor containing positional encoding and view direction encoding
               [batch_size, input_ch + input_ch_views]
               
        Returns:
            outputs: Output tensor containing RGB and density
                     [batch_size, output_ch]
        """
        # TODO: Implement the forward pass
        # 1. Split input into position and view components
        # 2. Process position through position MLP with skip connections
        # 3. If using viewdirs:
        #    - Compute alpha from current features
        #    - Process features and view directions through view MLP
        #    - Compute RGB from view-dependent features
        #    - Concatenate RGB and alpha as output
        # 4. If not using viewdirs, compute output directly from position features
        # 5. Return output
        
        pass  # Replace with your implementation

    def load_weights_from_keras(self, weights):
        """
        Load weights from a Keras checkpoint.
        
        Args:
            weights: Weights from Keras model
        """
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


# Ray helpers
def get_rays(H, W, K, c2w):
    """
    Generate camera rays given camera intrinsics and extrinsics.
    
    Args:
        H: Image height
        W: Image width
        K: Camera intrinsic matrix [3, 3]
        c2w: Camera-to-world transformation matrix [3, 4]
        
    Returns:
        rays_o: Ray origins [H, W, 3]
        rays_d: Ray directions [H, W, 3]
    """
    # TODO: Implement the get_rays function
    # 1. Create a meshgrid of pixel coordinates
    # 2. Convert pixel coordinates to camera coordinates using the intrinsic matrix
    # 3. Convert camera coordinates to world coordinates using the c2w matrix
    # 4. Set ray origins to the camera position (last column of c2w)
    # 5. Return ray origins and directions
    
    pass  # Replace with your implementation


def get_rays_np(H, W, K, c2w):
    """
    NumPy version of get_rays function.
    
    Args:
        H: Image height
        W: Image width
        K: Camera intrinsic matrix [3, 3]
        c2w: Camera-to-world transformation matrix [3, 4]
        
    Returns:
        rays_o: Ray origins [H, W, 3]
        rays_d: Ray directions [H, W, 3]
    """
    # TODO: Implement the get_rays_np function
    # Similar to get_rays but using NumPy operations instead of PyTorch
    
    pass  # Replace with your implementation


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world space to normalized device coordinates (NDC).
    
    Args:
        H: Image height
        W: Image width
        focal: Focal length
        near: Near plane distance
        rays_o: Ray origins in world space [batch_size, 3]
        rays_d: Ray directions in world space [batch_size, 3]
        
    Returns:
        rays_o: Ray origins in NDC [batch_size, 3]
        rays_d: Ray directions in NDC [batch_size, 3]
    """
    # TODO: Implement the ndc_rays function
    # 1. Shift ray origins to the near plane
    # 2. Project ray origins to NDC space
    # 3. Project ray directions to NDC space
    # 4. Return transformed ray origins and directions
    
    pass  # Replace with your implementation


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    Sample from a probability density function (PDF) using inverse transform sampling.
    
    This function is used for hierarchical sampling in the NeRF model.
    
    Args:
        bins: Bin edges for PDF [batch_size, N_bins]
        weights: Weights for each bin [batch_size, N_bins-1]
        N_samples: Number of samples to generate
        det: Whether to use deterministic sampling
        pytest: Whether to use fixed random numbers for testing
        
    Returns:
        samples: Sampled positions [batch_size, N_samples]
    """
    # TODO: Implement the sample_pdf function
    # 1. Add a small epsilon to weights to prevent NaNs
    # 2. Compute PDF by normalizing weights
    # 3. Compute CDF by cumulative sum of PDF
    # 4. Prepend a zero to CDF for the left boundary
    # 5. Generate uniform samples (deterministic or random)
    # 6. Use inverse transform sampling by finding sample locations in the CDF
    # 7. Interpolate between bin edges to get final samples
    # 8. Return samples
    
    pass  # Replace with your implementation