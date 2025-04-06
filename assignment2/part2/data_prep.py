"""
Data preparation script for creating training and testing pairs from the LFW dataset

This script creates a balanced set of same-person and different-person pairs for
training and testing a Siamese network for face verification.
"""

import os
import random
import argparse
from collections import defaultdict

def generate_pairs(lfw_dir, train_ratio=0.8, num_pairs=10000, same_ratio=0.5):
    """
    Generate pairs of face images for training and testing
    
    Args:
        lfw_dir: Directory containing LFW dataset
        train_ratio: Ratio of data to use for training (rest for testing)
        num_pairs: Total number of pairs to generate
        same_ratio: Ratio of same-person pairs (vs different-person pairs)
        
    Returns:
        train_pairs: List of training pairs (img1_path, img2_path, label)
        test_pairs: List of testing pairs (img1_path, img2_path, label)
    """
    print(f"Generating {num_pairs} pairs from LFW dataset in {lfw_dir}")
    
    # Group images by person
    person_images = defaultdict(list)
    for person_dir in os.listdir(lfw_dir):
        person_path = os.path.join(lfw_dir, person_dir)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(person_dir, img_name)
                    person_images[person_dir].append(img_path)
    
    # Filter out people with only one image (can't create same-person pairs)
    multi_img_people = {person: images for person, images in person_images.items() if len(images) > 1}
    
    if len(multi_img_people) == 0:
        raise ValueError("No people with multiple images found in the dataset")
    
    all_people = list(person_images.keys())
    people_with_multiple = list(multi_img_people.keys())
    
    # Calculate number of same and different pairs
    num_same = int(num_pairs * same_ratio)
    num_diff = num_pairs - num_same
    
    # Create same person pairs
    same_pairs = []
    for _ in range(num_same):
        person = random.choice(people_with_multiple)
        # Choose two different images of the same person
        img1, img2 = random.sample(multi_img_people[person], 2)
        same_pairs.append((img1, img2, 0))  # 0 indicates same person
    
    # Create different person pairs
    diff_pairs = []
    for _ in range(num_diff):
        # Choose two different people
        person1, person2 = random.sample(all_people, 2)
        img1 = random.choice(person_images[person1])
        img2 = random.choice(person_images[person2])
        diff_pairs.append((img1, img2, 1))  # 1 indicates different people
    
    # Combine and shuffle all pairs
    all_pairs = same_pairs + diff_pairs
    random.shuffle(all_pairs)
    
    # Split into training and testing sets
    train_size = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:train_size]
    test_pairs = all_pairs[train_size:]
    
    print(f"Generated {len(train_pairs)} training pairs and {len(test_pairs)} testing pairs")
    print(f"Same-person pairs: {num_same}, Different-person pairs: {num_diff}")
    
    return train_pairs, test_pairs

def write_pairs_to_file(pairs, output_file):
    """
    Write pairs to a text file
    
    Args:
        pairs: List of tuples (img1_path, img2_path, label)
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        for img1, img2, label in pairs:
            f.write(f"{img1} {img2} {label}\n")
    print(f"Wrote {len(pairs)} pairs to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate pairs for Siamese network training")
    parser.add_argument("--lfw_dir", type=str, default="./lfw", 
                        help="Directory containing LFW dataset")
    parser.add_argument("--train_file", type=str, default="./train.txt", 
                        help="Output file for training pairs")
    parser.add_argument("--test_file", type=str, default="./test.txt", 
                        help="Output file for testing pairs")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                        help="Ratio of data to use for training")
    parser.add_argument("--num_pairs", type=int, default=10000, 
                        help="Total number of pairs to generate")
    parser.add_argument("--same_ratio", type=float, default=0.5, 
                        help="Ratio of same-person pairs")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Generate pairs
    train_pairs, test_pairs = generate_pairs(
        args.lfw_dir, 
        args.train_ratio, 
        args.num_pairs, 
        args.same_ratio
    )
    
    # Write pairs to files
    write_pairs_to_file(train_pairs, args.train_file)
    write_pairs_to_file(test_pairs, args.test_file)

if __name__ == "__main__":
    main()