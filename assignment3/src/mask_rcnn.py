import torch.nn as nn
import torchvision
import torch
from .faster_rcnn import FasterRCNN, transform_boxes_to_original_size, RegionProposalNetwork

import torch
import torch.nn as nn
import torchvision

class MaskRCNN(nn.Module):
    """Mask R-CNN object detection and segmentation model"""
    
    def __init__(self, model_config, num_classes):
        super(MaskRCNN, self).__init__()
        self.model_config = model_config
        
        # VGG16 backbone (same as Faster R-CNN)
        vgg16 = torchvision.models.vgg16(weights="DEFAULT")
        self.backbone = vgg16.features[:-1]
        
        # Initialize the RPN (same as Faster R-CNN)
        self.rpn = RegionProposalNetwork(
            in_channels=512,
            scales=model_config['scales'],
            aspect_ratios=model_config['aspect_ratios'],
            model_config=model_config
        )
        
        # Replace ROI head with Mask head, same args pattern
        self.mask_head = MaskHead(
            num_classes=num_classes,
            in_channels=512,
            model_config=model_config
        )
        
        # Freeze early backbone layers (same as Faster R-CNN)
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        
        # Image normalization parameters (same as Faster R-CNN)
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']
    
    def normalize_resize_image_and_boxes(self, image, bboxes=None):
        """Normalize and resize image, adjusting bboxes accordingly"""
        # SAME implementation as Faster R-CNN
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        c, h, w = image.shape[-3:]
        
        image = image.float()
        mean = torch.as_tensor(self.image_mean, dtype=image.dtype, device=image.device)
        std = torch.as_tensor(self.image_std, dtype=image.dtype, device=image.device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        
        min_original_size = float(min((h, w)))
        max_original_size = float(max((h, w)))
        scale_factor_min = self.min_size / min_original_size
        
        if max_original_size * scale_factor_min > self.max_size:
            scale_factor = self.max_size / max_original_size
        else:
            scale_factor = scale_factor_min
        
        image = torch.nn.functional.interpolate(
            image, scale_factor=scale_factor, mode='bilinear',
            recompute_scale_factor=True, align_corners=False
        )
        
        if bboxes is not None:
            if bboxes.dim() == 2:
                ratios = [
                    torch.tensor(s, dtype=torch.float32, device=bboxes.device) /
                    torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                    for s, s_orig in zip(image.shape[-2:], (h, w))
                ]
                ratio_height, ratio_width = ratios
                
                xmin = bboxes[:, 0] * ratio_width
                ymin = bboxes[:, 1] * ratio_height
                xmax = bboxes[:, 2] * ratio_width
                ymax = bboxes[:, 3] * ratio_height
                
                bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
            
            elif bboxes.dim() == 3:
                ratios = [
                    torch.tensor(s, dtype=torch.float32, device=bboxes.device) /
                    torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                    for s, s_orig in zip(image.shape[-2:], (h, w))
                ]
                ratio_height, ratio_width = ratios
                
                xmin, ymin, xmax, ymax = bboxes.unbind(2)
                xmin = xmin * ratio_width
                xmax = xmax * ratio_width
                ymin = ymin * ratio_height
                ymax = ymax * ratio_height
                
                bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
        
        return image, bboxes
    
    def forward(self, image, target=None):
        """Forward pass for Mask R-CNN"""
        old_shape = image.shape[-2:]
        if self.training:
            image, bboxes = self.normalize_resize_image_and_boxes(
                image, target['bboxes']
            )
            target['bboxes'] = bboxes
        else:
            image, _ = self.normalize_resize_image_and_boxes(image, None)
        
        feat = self.backbone(image)
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']
        
        # Use mask_head instead of roi_head
        maskrcnn_output = self.mask_head(feat)
        
        if not self.training:
            # print(f"Type of maskrcnn_output: {type(maskrcnn_output)}")
            # print(f"maskedrcnn_output['boxes']", maskrcnn_output['boxes'])
            # boxes = maskrcnn_output['boxes']
            boxes = maskrcnn_output
            
            # Example fix: remove extra dims if needed (adjust as needed after print debug)
            if boxes.dim() > 2:
                boxes = boxes.view(-1, 4)  # flatten to [num_boxes, 4]

            maskrcnn_output = transform_boxes_to_original_size(
                boxes,
                image.shape[-2:],
                old_shape
            )
        
        return rpn_output, maskrcnn_output

class MaskHead(nn.Module):
    def __init__(self, in_channels, num_classes, model_config):
        super().__init__()
        # Standard mask head: conv layers followed by upsampling and 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.mask_predictor = nn.Conv2d(256, num_classes, kernel_size=1)
        self.model_config = model_config
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.deconv(x))
        x = self.mask_predictor(x)
        return x

    def loss(self, mask_preds, gt_masks, proposals, gt_classes):
        """
        mask_preds: [num_proposals, num_classes, 14, 14]
        gt_masks: list of tensors [num_pos, H, W] or [num_pos, 1, H, W] masks for positive proposals
        proposals: list or tensor of proposals (num_proposals)
        gt_classes: tensor of ground-truth classes for positive proposals (num_pos)
        """

        # For this example, assume all proposals are positive (or filter them before)
        # So num_proposals == num_pos

        # Select predicted masks for the gt classes
        idx = torch.arange(mask_preds.shape[0], device=mask_preds.device)
        pred_masks_for_gt_classes = mask_preds[idx, gt_classes]  # shape: [num_pos, 14, 14]

        # Resize gt_masks to 14x14
        if len(gt_masks.shape) == 4:
            gt_masks = gt_masks.squeeze(1)  # remove channel if present

        gt_masks_resized = F.interpolate(gt_masks.unsqueeze(1).float(), size=(14, 14), mode='bilinear', align_corners=False)
        gt_masks_resized = gt_masks_resized.squeeze(1)  # [num_pos, 14, 14]

        # Compute binary cross entropy with logits
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(pred_masks_for_gt_classes, gt_masks_resized)

        return loss
