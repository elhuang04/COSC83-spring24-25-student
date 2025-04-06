I'll update the README based on your requirements, focusing on a textual description of the network architecture, making BCE the default loss function, and adding tips about using generative AI and Weights & Biases for visualization.

# Computer Vision Assignment: Feature Detection, Matching, and Deep Learning Methods

## Overview
This assignment will give you hands-on experience with both traditional computer vision techniques and deep learning approaches. The assignment consists of two parts: (1) traditional methods for feature detection, description, and matching, and (2) Siamese networks for image matching.

## Learning Objectives
- Understand and implement the Harris corner detector
- Learn about scale and rotation invariant feature descriptors (SIFT/SURF)
- Implement RANSAC for robust feature matching
- Design and train Siamese networks for image matching
- Compare traditional and deep learning approaches
- Visualize matching results and evaluate matching quality

## Assignment Structure

### Part 1: Traditional Methods (60%)

#### Part 1A: Harris Corner Detector
Implement the Harris corner detector to identify interest points in images.

**Requirements:**
- Implement gradient computation using Sobel operators
- Compute the structure tensor for each pixel
- Apply non-maximum suppression to find corner points
- Allow parameter adjustment (corner response threshold, window size)
- Visualize detected corners on the original images

#### Part 1B: Feature Description (SIFT/SURF)
Use OpenCV's implementation of SIFT or SURF for feature description.

**Requirements:**
- Use OpenCV's SIFT or SURF implementation to extract feature descriptors
- Experiment with different parameter settings and analyze their effects
- Visualize keypoints with orientation and scale information
- Compare performance between SIFT and SURF in terms of accuracy and speed
- Understand and document the underlying principles of the chosen descriptor

#### Part 1C: Feature Matching with RANSAC
Implement feature matching between image pairs and use RANSAC to find the best geometric transformation.

**Requirements:**
- Implement a distance metric for descriptor matching (e.g., Euclidean distance)
- Apply ratio test to filter out ambiguous matches
- Implement RANSAC to find a robust homography matrix
- Visualize matches between image pairs (before and after RANSAC)
- Implement a ranking metric to evaluate matching quality between image pairs
- Create a leaderboard showing the best and worst matching image pairs
- ***Please get 1 more data from the panaroma dataset other than the provided***.

here are some demo for the results
### Results Table with Figures

| **Task**                  | **Description**                                   | **Figure**                                                                 |
|---------------------------|---------------------------------------------------|----------------------------------------------------------------------------|
| Harris Corner Detection   | Visualization of detected corners on sample image | ![Harris Corners](part1/results/img1_img2/harris_corners1.jpg)                              |
| SIFT Keypoints            | Keypoints detected using SIFT with orientation    | ![SIFT Keypoints](part1/results/img1_img2/sift_keypoints1.jpg)                              |
| Feature Matching (Before) | Matches before applying RANSAC                   | ![Feature Matching Before](part1/results/img1_img2/initial_matches.jpg)            |
| Feature Matching (After)  | Matches after applying RANSAC                    | ![Feature Matching After](part1/results/img1_img2/ransac_matches.jpg)              |


### Part 2: Siamese Networks for Image Matching (40%)

Design and implement a Siamese network for image matching using deep learning techniques.

**Requirements:**
- Implement the architecture of a Siamese Network for image matching
- Train the network on image pairs using Binary Cross-Entropy (BCE) loss
- Evaluate the model on unseen image pairs
- Compare the deep learning approach to traditional feature matching methods

## Tasks to Complete

### Part 1: Traditional Methods
Your task is to complete the implementation in the following files:

1. In `src/harris.py`:
   - Implement gradient computation using Sobel operators
   - Compute the structure tensor
   - Calculate Harris corner response
   - Apply non-maximum suppression

2. In `src/descriptors.py`:
   - Initialize SIFT/SURF feature descriptor from OpenCV
   - Implement keypoint detection and descriptor computation
   - Convert Harris corners to keypoints

3. In `src/matching.py`:
   - Implement descriptor matching with Lowe's ratio test
   - Implement RANSAC for homography estimation
   - Compute match quality score

### Part 2: Siamese Networks
Your task is to implement the Siamese network architecture and its training loop:

1. In `part2/model.py`:
   - Implement a Siamese Network architecture using ResNet18 as the backbone
   - The network should output a similarity score for image pairs using BCE loss

2. In `part2/main.py`:
   - Implement the training loop
   - Set up data loading and device placement
   - Implement forward pass, loss calculation, and backpropagation
   - Track statistics and periodically evaluate the model
   - Save checkpoints and visualize results

## Siamese Network Architecture Description

Your Siamese Network implementation should follow this architecture:

1. **Feature Extraction Backbone**: 
   - Use ResNet18 as the feature extraction backbone
   - Modify the first convolutional layer to handle RGB images (3 channels)
   - Remove the final classification layer to extract feature vectors

2. **Feature Processing Pathway**:
   - Process each input image through the same backbone network
   - Extract feature vectors for both input images
   - Concatenate the two feature vectors

3. **Similarity Prediction**:
   - Add fully connected layers after concatenation (input_size*2 → 256 → 1)
   - Apply ReLU activation after the first fully connected layer
   - Apply Sigmoid activation at the output to get a similarity score (0-1)

4. **Weight Initialization**:
   - Initialize weights with Xavier initialization
   - Initialize biases with small positive values

## Training and Evaluation Tips

1. **Data Loading and Preprocessing**:
   - Load image pairs from the dataset
   - Apply appropriate transformations (resize, normalization)
   - Create balanced batches of similar and dissimilar pairs

2. **Training Loop Implementation**:
   - Iterate through epochs and batches
   - Move data to the appropriate device (CPU/GPU)
   - Perform forward pass to get similarity predictions
   - Calculate BCE loss between predictions and ground truth
   - Perform backward pass and optimization
   - Track and log metrics

3. **Visualization and Logging**:(You can use GenAI to do this)
   - Use Weights & Biases (wandb) to track and visualize training progress (*Recommend to use this which will be really helpful in your future research*)
   - Log metrics such as loss, accuracy, precision, and recall
   - Visualize example matches and mismatches periodically


## Bonus (10%)
You will get full marks for bonus if you finish the analysis of using Contrastive Loss, the Triplet loss is optional, but you should take a lot at it.
1. **Contrastive Loss Implementation**:
   - Use the Contrastive loss to train the model
   - Please make sure you fully understand it, write your understanding on the reason why it works better than BCE Loss
2. **Triplet Loss Implementation**:
   - Implement triplet loss as an alternative approach
   - Create triplets of anchor, positive, and negative samples
   - Compare results across all three loss functions

## Dataset
The assignment will use a dataset consisting of:
- Image pairs with varying degrees of viewpoint change. Please also find more data to test from [Kaggle's panorama dataset](https://www.kaggle.com/datasets/yaseenksk/dataset-panorama).
- Oxford5k (https://www.kaggle.com/datasets/vadimshabashov/oxford5k/data?select=groundtruth.json) for face matching experiments. But provided on Canvas

## Implementation Details
- Programming language: Python
- Required libraries: NumPy, OpenCV, Matplotlib, PyTorch, wandb
- You must implement the Harris corner detector and RANSAC algorithm yourself
- You may use OpenCV for SIFT/SURF feature extraction
- You should implement the Siamese network architecture yourself

## Testing Your Implementation

### Part 1: Traditional Methods
You can test your implementation using the provided test script:

```
python test_assignment.py
```

This will run tests for each component and save the results as images.

Once you're confident in your implementation, you can run the full pipeline:

```
python main.py
```

### Part 2: Siamese Networks
Test your implementation with:

```
python main.py --action=train_test
```

## Running the Assignment

1. Download the Oxford5K dataset provided on Canvas and place your image pairs in the `images` directory for part1.
2. Run the main script for traditional methods:
   ```
   python main.py
   ```
3. Run the training and evaluation scripts for Siamese networks in part2:
   ```
   python main.py --action=train_test
   ```
4. Check the `results` directory for the output visualizations

## Evaluation Criteria
Your assignment will be evaluated based on:

### Part 1: Traditional Methods (60%)
- Correctness of implementation for Harris detector (20%)
- Appropriate use and understanding of SIFT/SURF (20%)
- Quality of feature matching and RANSAC implementation (20%)

### Part 2: Siamese Networks (40%)
- Correctness of implementation for Siamese network architecture (20%)
- Completeness and correctness of training loop (10%)
- Analysis of results and performance (10%)

## Submission Requirements
1. Source code with clear documentation
2. LaTeX report (PDF) containing:
   - Brief description of your implementation for both parts
   - Visualization of results at each stage
   - Analysis of Siamese network training 
   - Comparison between traditional methods and Siamese network
   - Analysis of matching performance across different image pairs
   - Discussion of limitations and potential improvements
   - All experimental results with appropriate figures and tables

## Timeline
- Release date: [Date]
- Submission deadline: [Date]

## Resources
- Lecture notes on feature detection and matching
- Original papers:
  - Aggregating Deep Convolutional Features for Image Retrieval https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Babenko_Aggregating_Local_Deep_ICCV_2015_paper.pdf
  - Harris & Stephens, "A combined corner and edge detector" (1988)
  - Lowe, "Distinctive Image Features from Scale-Invariant Keypoints" (2004)
  - Bay et al., "SURF: Speeded Up Robust Features" (2006)
  - Fischler & Bolles, "Random Sample Consensus" (1981)
  - Bromley et al., "Signature Verification using a Siamese Time Delay Neural Network" (1993)
  - Chopra et al., "Learning a Similarity Metric Discriminatively, with Application to Face Verification" (2005)
- OpenCV documentation for SIFT and SURF implementations
- PyTorch documentation for deep learning implementation
- Weights & Biases documentation for experiment tracking

## Questions and Support
If you have any questions about the assignment, please contact the teaching assistants during office hours.