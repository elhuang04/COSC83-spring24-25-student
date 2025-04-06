"""
Dataset handling for feature matching with Oxford/All Souls dataset
"""
import os
import random
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, random_split
import itertools
import json

class FeatureMatchingDataset(Dataset):
    """
    Dataset class for loading and preprocessing image pairs for feature matching
    
    Args:
        root_dir (string): Directory with all the images (contains 'images' folder)
        data_name (string): Path to the JSON data file with image information
        transform (callable, optional): Optional transform to be applied on a sample
        random_aug (bool): Whether to apply random augmentation
        categories (list): List of categories to include (default: ["good", "ok", "junk", "query"])
        max_pairs_per_location (int): Maximum number of positive pairs to generate per location
        max_neg_pairs_per_location_pair (int): Maximum number of negative pairs to generate per location pair
        split (string): 'train', 'test', or None (full dataset)
        test_ratio (float): Ratio of data to use for test split (0.0-1.0)
        seed (int): Random seed for reproducibility
    """
    def __init__(self, root_dir, data_name, transform=None, random_aug=False, 
                 categories=["good", "ok", "junk", "query"],
                 max_pairs_per_location=200,
                 max_neg_pairs_per_location_pair=100,
                 split=None,
                 test_ratio=0.2,
                 seed=42):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.data_name = data_name
        self.categories = categories
        self.max_pairs_per_location = max_pairs_per_location
        self.max_neg_pairs_per_location_pair = max_neg_pairs_per_location_pair
        self.split = split
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Parse the data structure from data_name
        with open(data_name, 'r') as f:
            data_dict = json.load(f)
        
        # Generate all image pairs
        all_pairs = self._generate_pairs(data_dict)
        
        # Apply train/test split if requested
        if split is not None:
            train_pairs, test_pairs = self._train_test_split(all_pairs, test_ratio, seed)
            
            if split == 'train':
                self.data = train_pairs
                print(f"Using training split ({len(self.data)} pairs)")
            elif split == 'test':
                self.data = test_pairs
                print(f"Using test split ({len(self.data)} pairs)")
            else:
                raise ValueError(f"Invalid split value: {split}. Must be 'train', 'test', or None")
        else:
            self.data = all_pairs
            print(f"Using full dataset ({len(self.data)} pairs)")
        
        # Print dataset statistics
        self._print_statistics()
        
        self.transform = transform
        self.random_aug = random_aug
        self.random_aug_prob = 0.7

    def _generate_pairs(self, data_dict):
        """Generate all image pairs from the data dictionary"""
        all_pairs = []
        
        # Extract locations (all_souls, pitt_rivers, hertford, etc.)
        locations = data_dict.keys()
        
        # For each location, collect all images across all categories
        location_images = {}
        for location in locations:
            location_images[location] = []
            for category in self.categories:
                if category in data_dict[location]:
                    location_images[location].extend(data_dict[location][category])
        
        # Create positive pairs (same location, any category)
        for location, images in location_images.items():
            # If there are too many possible pairs, sample randomly
            if len(images) > 20:  # Arbitrary threshold to decide when to sample
                pairs_to_generate = min(self.max_pairs_per_location, len(images) * (len(images) - 1) // 2)
                
                # Generate random pairs
                pairs_generated = 0
                attempts = 0
                max_attempts = pairs_to_generate * 10  # Avoid infinite loop
                
                while pairs_generated < pairs_to_generate and attempts < max_attempts:
                    attempts += 1
                    i, j = random.sample(range(len(images)), 2)
                    if i != j:
                        img1 = images[i]
                        img2 = images[j]
                        # Label 1 for positive pair (same location)
                        all_pairs.append((img1, img2, 1, location))
                        pairs_generated += 1
            else:
                # For small number of images, create all possible pairs
                for i in range(len(images)):
                    for j in range(i+1, len(images)):
                        img1 = images[i]
                        img2 = images[j]
                        # Label 1 for positive pair (same location)
                        all_pairs.append((img1, img2, 1, location))
        
        # Create negative pairs (different locations)
        location_list = list(locations)
        for i, loc1 in enumerate(location_list):
            for j, loc2 in enumerate(location_list[i+1:], i+1):
                images1 = location_images[loc1]
                images2 = location_images[loc2]
                
                if not images1 or not images2:
                    continue
                    
                # Determine number of pairs to generate
                num_pairs = min(self.max_neg_pairs_per_location_pair, 
                               len(images1) * len(images2) // 10)  # Limit to 10% of possible pairs
                
                # Generate random pairs
                for _ in range(num_pairs):
                    img1 = random.choice(images1)
                    img2 = random.choice(images2)
                    # Label 0 for negative pair (different locations)
                    all_pairs.append((img1, img2, 0, f"{loc1}_{loc2}"))
        
        return all_pairs

    def _train_test_split(self, pairs, test_ratio, seed):
        """Split pairs into training and testing sets"""
        # Group pairs by location and label
        groups = {}
        for pair in pairs:
            _, _, label, location_info = pair
            key = f"{location_info}_{label}"
            if key not in groups:
                groups[key] = []
            groups[key].append(pair)
        
        # Split each group to maintain distribution
        train_pairs = []
        test_pairs = []
        
        for group_name, group_pairs in groups.items():
            random.shuffle(group_pairs)  # Shuffle pairs within each group
            
            test_size = int(len(group_pairs) * test_ratio)
            train_size = len(group_pairs) - test_size
            
            train_pairs.extend(group_pairs[:train_size])
            test_pairs.extend(group_pairs[train_size:])
        
        # Shuffle final lists
        random.shuffle(train_pairs)
        random.shuffle(test_pairs)
        
        return train_pairs, test_pairs

    def _print_statistics(self):
        """Print dataset statistics"""
        print(f"Dataset split: {self.split if self.split else 'full'}")
        print(f"Total pairs: {len(self.data)}")
        print(f"Positive pairs: {sum(1 for _, _, label, _ in self.data if label == 1)}")
        print(f"Negative pairs: {sum(1 for _, _, label, _ in self.data if label == 0)}")
        
        # Print pairs per location
        pos_pairs_by_loc = {}
        for _, _, label, pair_info in self.data:
            if label == 1:
                pos_pairs_by_loc[pair_info] = pos_pairs_by_loc.get(pair_info, 0) + 1
        
        if pos_pairs_by_loc:
            print("Positive pairs by location:")
            for loc, count in pos_pairs_by_loc.items():
                print(f"  {loc}: {count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1_file, img2_file, label, pair_info = self.data[idx]
        
        # Remove quotes if present in the filename
        img1_file = img1_file.strip('"')
        img2_file = img2_file.strip('"')
        
        # Load images
        img1_path = os.path.join(self.images_dir, img1_file)
        img2_path = os.path.join(self.images_dir, img2_file)
        
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}")
            print(f"Paths: {img1_path}, {img2_path}")
            # Return a placeholder in case of error
            img1 = Image.new('RGB', (256, 256))
            img2 = Image.new('RGB', (256, 256))
            label = 0
        
        # Apply random augmentation if specified
        if self.random_aug and random.random() < self.random_aug_prob:
            img1 = self.random_augmentation(img1)
            # For positive pairs, apply similar augmentation to create correspondence
            if label == 1:
                img2 = self.random_augmentation(img2)
            # For negative pairs, apply different augmentation
            else:
                img2 = self.random_augmentation(img2)
        
        # Apply transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Convert imgs to tensors
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        return (img1, img2, label)

    def random_augmentation(self, img):
        """
        Apply one or more of the following augmentations randomly:
        - Rotation
        - Horizontal flipping
        - Translation
        - Scaling
        
        These augmentations are particularly useful for feature matching
        as they simulate different viewpoints of the same structure.
        """
        def rotate(img):
            # For buildings, smaller rotations may be more appropriate
            degree = random.randrange(-15, 15)
            return img.rotate(degree)
        
        def flip(img):
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        
        def translate(img):
            d_x = random.randrange(-20, 20)
            d_y = random.randrange(-20, 20)
            img = np.array(img)
            mat = np.float32([[1, 0, d_x], [0, 1, d_y]])
            num_rows, num_cols = img.shape[:2]
            img = cv2.warpAffine(img, mat, (num_cols, num_rows))
            return Image.fromarray(np.uint8(img))
        
        def scale(img):
            # More significant scale variations for buildings
            scale = 0.8 + 0.4 * random.random()
            img = np.array(img)
            num_rows, num_cols = img.shape[:2]
            # Keep the image centered while scaling
            tx = (num_cols * (1-scale)) / 2
            ty = (num_rows * (1-scale)) / 2
            mat = np.float32([[scale, 0, tx], [0, scale, ty]])
            img = cv2.warpAffine(img, mat, (num_cols, num_rows))
            return Image.fromarray(np.uint8(img))
        
        # Apply random transformations
        transform_ops = [rotate, flip, translate, scale]
        op_len = random.randrange(1, len(transform_ops) + 1)
        ops = random.sample(transform_ops, op_len)
        for op in ops:
            img = op(img)

       
        
        return img


def create_train_test_datasets(root_dir, data_name, transform=None, random_aug=False, 
                               categories=["good", "ok", "junk", "query"],
                               max_pairs_per_location=200,
                               max_neg_pairs_per_location_pair=100,
                               test_ratio=0.2,
                               seed=42):
    """
    Helper function to create train and test datasets with the same split
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    train_dataset = FeatureMatchingDataset(
        root_dir=root_dir,
        data_name=data_name,
        transform=transform,
        random_aug=random_aug,
        categories=categories,
        max_pairs_per_location=max_pairs_per_location,
        max_neg_pairs_per_location_pair=max_neg_pairs_per_location_pair,
        split='train',
        test_ratio=test_ratio,
        seed=seed
    )
    
    test_dataset = FeatureMatchingDataset(
        root_dir=root_dir,
        data_name=data_name,
        transform=transform,
        random_aug=random_aug,
        categories=categories,
        max_pairs_per_location=max_pairs_per_location,
        max_neg_pairs_per_location_pair=max_neg_pairs_per_location_pair,
        split='test',
        test_ratio=test_ratio,
        seed=seed
    )
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    dataset = FeatureMatchingDataset(root_dir='./', 
                                     data_name='groundtruth.json', split='Train')
    print(f"Number of samples: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample: {sample}")
    print(f"Label: {sample[2]}")