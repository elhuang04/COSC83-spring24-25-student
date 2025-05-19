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
from src.mask_rcnn import MaskRCNN
from dataset.voc import VOCDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('step_count', 1)

def train(args):
    # Load config file
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
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

    voc = VOCDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'],
                     use_masks=True)
    
    train_loader = DataLoader(voc,
                              batch_size=1,
                              shuffle=True,
                              num_workers=4)
    
    model = MaskRCNN(model_config, num_classes=dataset_config['num_classes'])
    model.train()
    model.to(device)

    os.makedirs(train_config['task_name'], exist_ok=True)
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])

    optimizer = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_config['lr'],
        momentum=0.9,
        weight_decay=5e-4
    )
    
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']

    start_epoch = 0
    step_count = 1

    # Load checkpoint if it exists
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        start_epoch, step_count = load_checkpoint(ckpt_path, model, optimizer, scheduler)
        print(f"Resuming from epoch {start_epoch}, step {step_count}")

    for epoch in range(start_epoch, num_epochs):
        rpn_cls_losses = []
        rpn_loc_losses = []
        mask_losses = []
        optimizer.zero_grad()

        for im, target, fname in tqdm(train_loader):
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            target['masks'] = target['masks'].float().to(device)

            rpn_output, maskrcnn_output = model(im, target)

            rpn_cls_loss = rpn_output['rpn_classification_loss']
            rpn_loc_loss = rpn_output['rpn_localization_loss']

            mask_preds = maskrcnn_output
            proposals = rpn_output['proposals']
            gt_classes = target['labels']
            gt_masks = target['masks']

            mask_loss = model.mask_head.loss(mask_preds, gt_masks, proposals, gt_classes)

            total_loss = rpn_cls_loss + rpn_loc_loss + mask_loss

            rpn_cls_losses.append(rpn_cls_loss.item())
            rpn_loc_losses.append(rpn_loc_loss.item())
            mask_losses.append(mask_loss.item())

            (total_loss / acc_steps).backward()

            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1

        print(f"Finished epoch {epoch}")
        optimizer.step()
        optimizer.zero_grad()

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'step_count': step_count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, ckpt_path)

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
