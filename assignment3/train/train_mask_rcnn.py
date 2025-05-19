import torch
import argparse
import os
import numpy as np
import yaml
import random
import sys
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mask_rcnn import MaskRCNN  # Your MaskRCNN implementation
from dataset.voc import VOCDataset   # Assuming your dataset also provides masks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # Load config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # Your VOCDataset should return image, target (dict with bboxes, labels, masks), filename
    voc = VOCDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'],
                     use_masks=True)  # <- ensure dataset supports masks
    
    train_loader = DataLoader(voc,
                              batch_size=1,
                              shuffle=True,
                              num_workers=4)
    
    model = MaskRCNN(model_config, num_classes=dataset_config['num_classes'])
    model.train()
    model.to(device)

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    optimizer = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_config['lr'],
        momentum=0.9,
        weight_decay=5e-4
    )
    
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1

    for epoch in range(num_epochs):
        rpn_cls_losses = []
        rpn_loc_losses = []
        mask_losses = []
        optimizer.zero_grad()

        for im, target, fname in tqdm(train_loader):
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            target['masks'] = target['masks'].float().to(device)  # Assuming masks are float tensors

            # Forward pass: returns dicts for rpn and mask heads
            rpn_output, maskrcnn_output = model(im, target)

            # Compute RPN losses (classification + localization)
            rpn_cls_loss = rpn_output['rpn_classification_loss']
            rpn_loc_loss = rpn_output['rpn_localization_loss']

            # Compute Mask Head loss using your MaskHead.loss method
            # Assuming maskrcnn_output is the mask predictions tensor
            mask_preds = maskrcnn_output  # [num_proposals, num_classes, 14, 14]

            # We need proposals and gt_classes for MaskHead.loss
            # For simplicity, assuming target includes these or you can modify accordingly
            proposals = rpn_output['proposals']
            gt_classes = target['labels']

            # Filter masks and proposals for positive samples (example, depends on your implementation)
            # Here, assume all targets are positives for demo
            gt_masks = target['masks']

            mask_loss = model.mask_head.loss(mask_preds, gt_masks, proposals, gt_classes)

            total_loss = rpn_cls_loss + rpn_loc_loss + mask_loss

            rpn_cls_losses.append(rpn_cls_loss.item())
            rpn_loc_losses.append(rpn_loc_loss.item())
            mask_losses.append(mask_loss.item())

            loss = total_loss / acc_steps
            loss.backward()

            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1

        print(f"Finished epoch {epoch}")
        optimizer.step()
        optimizer.zero_grad()

        torch.save(model.state_dict(),
                   os.path.join(train_config['task_name'], train_config['ckpt_name']))

        print(f"RPN Classification Loss: {np.mean(rpn_cls_losses):.4f} | "
              f"RPN Localization Loss: {np.mean(rpn_loc_losses):.4f} | "
              f"Mask Loss: {np.mean(mask_losses):.4f}")
        scheduler.step()
    print('Done Training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for Mask R-CNN training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc_maskrcnn.yaml', type=str)
    args = parser.parse_args()
    train(args)
