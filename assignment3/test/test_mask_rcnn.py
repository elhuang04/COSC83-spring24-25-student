import torch
import numpy as np
import cv2
import argparse
import random
import os
import yaml
from tqdm import tqdm
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Adding to sys.path:", project_root)
sys.path.append(project_root)

from src.mask_rcnn import MaskRCNN   # Replace with your MaskRCNN import
from dataset.voc import VOCDataset   # Assume same VOC dataset works
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)

def load_model_and_dataset(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc = VOCDataset('test', im_dir=dataset_config['im_test_path'], ann_dir=dataset_config['ann_test_path'])
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False)

    model = MaskRCNN(model_config, num_classes=dataset_config['num_classes'])
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'], train_config['ckpt_name']),
                                     map_location=device))

    return model, voc, test_dataset


def visualize_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Apply mask on the image with given color and transparency."""
    for c in range(3):
        image[:, :, c] = np.where(mask, 
                                  image[:, :, c] * (1 - alpha) + alpha * color[c], 
                                  image[:, :, c])
    return image


def infer(args):
    if not os.path.exists('samples'):
        os.mkdir('samples')

    model, voc, test_dataset = load_model_and_dataset(args)

    # Set low score threshold (adjust as needed or from config)
    model.roi_head.low_score_threshold = 0.7

    for sample_count in tqdm(range(10)):
        random_idx = random.randint(0, len(voc) - 1)
        im_tensor, target, fname = voc[random_idx]
        im_tensor = im_tensor.unsqueeze(0).float().to(device)

        # Load original image for visualization
        orig_im = cv2.imread(fname)
        orig_im_gt = orig_im.copy()

        # Draw ground truth boxes
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
            cv2.rectangle(orig_im_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = voc.idx2label[target['labels'][idx].item()]
            cv2.putText(orig_im_gt, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(f'samples/output_maskrcnn_gt_{sample_count}.png', orig_im_gt)

        with torch.no_grad():
            rpn_out, maskrcnn_out = model(im_tensor, None)

        boxes = maskrcnn_out['boxes'].cpu().numpy()
        labels = maskrcnn_out['labels'].cpu().numpy()
        scores = maskrcnn_out['scores'].cpu().numpy()
        masks = maskrcnn_out['masks'].cpu().numpy()  # Shape (N, 1, H, W)

        im_vis = orig_im.copy()

        for idx in range(len(boxes)):
            if scores[idx] < model.roi_head.low_score_threshold:
                continue

            x1, y1, x2, y2 = boxes[idx].astype(int)
            label = voc.idx2label[labels[idx]]
            score = scores[idx]

            # Draw box
            cv2.rectangle(im_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f'{label}: {score:.2f}'
            cv2.putText(im_vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Overlay mask
            mask = masks[idx, 0]
            mask = (mask > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(im_vis, dtype=np.uint8)
            color = [random.randint(0, 255) for _ in range(3)]
            colored_mask[:, :, 0] = color[0]
            colored_mask[:, :, 1] = color[1]
            colored_mask[:, :, 2] = color[2]

            # Blend mask onto image
            im_vis = np.where(mask[:, :, None], 
                              (0.4 * colored_mask + 0.6 * im_vis).astype(np.uint8),
                              im_vis)

        cv2.imwrite(f'test_maskrcnn_output_{sample_count}.png', im_vis)


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate',
                        default=False, type=bool)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=True, type=bool)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.infer_samples:
        infer(args)
    else:
        print('Not Inferring for samples as `infer_samples` argument is False')

if __name__ == '__main__':
    main()
