import cv2
import numpy as np
import os

def warp_and_align_images(img1, img2, H, output_dir=None, pair_name=""):
    """
    Warp img1 to align with img2 using homography H.

    Args:
        img1 (np.ndarray): Source image.
        img2 (np.ndarray): Destination image.
        H (np.ndarray): Homography matrix.
        output_dir (str): Directory to save results.
        pair_name (str): Name of the image pair.

    Returns:
        aligned_img (np.ndarray): Warped version of img1.
        blend (np.ndarray): Blended image for visualization.
    """
    aligned_img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    blend = cv2.addWeighted(aligned_img, 0.5, img2, 0.5, 0)

    if output_dir and pair_name:
        cv2.imwrite(os.path.join(output_dir, f"{pair_name}_aligned.jpg"), aligned_img)
        cv2.imwrite(os.path.join(output_dir, f"{pair_name}_blend.jpg"), blend)

    return aligned_img, blend
