import torch
import torch.nn as nn
import torchvision
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ======================================================================
# Utility Functions (Already Implemented)
# ======================================================================

def get_iou(boxes1, boxes2):
    """Compute IoU between box sets (N x 4) and (M x 4)"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    
    # Compute intersection coordinates
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # (N, M)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # (N, M)
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # (N, M)
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # (N, M)
    
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)  # (N, M)
    union = area1[:, None] + area2 - intersection_area  # (N, M)
    iou = intersection_area / union  # (N, M)
    return iou


def boxes_to_transformation_targets(ground_truth_boxes, anchors_or_proposals):
    """Convert bbox coordinates to regression targets (tx, ty, tw, th)"""
    # Get center_x, center_y, w, h for anchors/proposals
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * widths
    center_y = anchors_or_proposals[:, 1] + 0.5 * heights

    # Get center_x, center_y, w, h for gt boxes
    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights

    # Compute regression targets
    targets_dx = (gt_center_x - center_x) / widths
    targets_dy = (gt_center_y - center_y) / heights
    targets_dw = torch.log(gt_widths / widths)
    targets_dh = torch.log(gt_heights / heights)

    regression_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return regression_targets

def apply_regression_pred_to_anchors_or_proposals(box_transform_pred, anchors_or_proposals):
    """Apply predicted transformations to anchors/proposals to get predicted boxes"""
    box_transform_pred = box_transform_pred.reshape(box_transform_pred.size(0), -1, 4)
    
    # Get cx, cy, w, h from x1, y1, x2, y2
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * w
    center_y = anchors_or_proposals[:, 1] + 0.5 * h
    
    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]
    
    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))
    
    # Apply transformations
    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]
    
    # Convert back to box coordinates
    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h
    
    pred_boxes = torch.stack((pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2), dim=2)
    return pred_boxes


def sample_positive_negative(labels, positive_count, total_count):
    """Sample positive and negative examples for training"""
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]
    
    # Cap number of positives and negatives
    num_pos = min(positive.numel(), positive_count)
    num_neg = min(negative.numel(), total_count - num_pos)
    
    # Random sampling
    perm_positive_idxs = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm_negative_idxs = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
    
    pos_idxs = positive[perm_positive_idxs]
    neg_idxs = negative[perm_negative_idxs]
    
    # Create masks
    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_pos_idx_mask[pos_idxs] = True
    sampled_neg_idx_mask[neg_idxs] = True
    
    return sampled_neg_idx_mask, sampled_pos_idx_mask


def clamp_boxes_to_image_boundary(boxes, image_shape):
    """Clip boxes to stay within image boundaries"""
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]
    
    height, width = image_shape[-2:]
    boxes_x1 = boxes_x1.clamp(min=0, max=width)
    boxes_x2 = boxes_x2.clamp(min=0, max=width)
    boxes_y1 = boxes_y1.clamp(min=0, max=height)
    boxes_y2 = boxes_y2.clamp(min=0, max=height)
    
    boxes = torch.cat((
        boxes_x1[..., None],
        boxes_y1[..., None],
        boxes_x2[..., None],
        boxes_y2[..., None]),
        dim=-1)
    return boxes


def transform_boxes_to_original_size(boxes, new_size, original_size):
    """Scale bounding boxes back to original image dimensions"""
    ratios = [
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) / 
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


# ======================================================================
# Part 2: Region Proposal Network (20%)
# ======================================================================

class RegionProposalNetwork(nn.Module):
    """Region Proposal Network for Faster R-CNN"""
    
    def __init__(self, in_channels, scales, aspect_ratios, model_config):
        super(RegionProposalNetwork, self).__init__()
        # Store configuration
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.low_iou_threshold = model_config['rpn_bg_threshold']
        self.high_iou_threshold = model_config['rpn_fg_threshold']
        self.rpn_nms_threshold = model_config['rpn_nms_threshold']
        self.rpn_batch_size = model_config['rpn_batch_size']
        self.rpn_pos_count = int(model_config['rpn_pos_fraction'] * self.rpn_batch_size)
        self.rpn_topk = model_config['rpn_train_topk'] if self.training else model_config['rpn_test_topk']
        self.rpn_prenms_topk = model_config['rpn_train_prenms_topk'] if self.training else model_config['rpn_test_prenms_topk']
        
        # DONE: Calculate the number of anchors per location based on scales and aspect ratios
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)
    
        # DONE: Implement the network layers
        # 1. A 3x3 convolutional layer with in_channels input channels and in_channels output channels
        
        self.rpn_conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size = 3,
                                  stride = 1,
                                  padding = 1)
        
        # 2. Two 1x1 convolutional layers for classification and regression
        #    - Classification layer should output num_anchors channels (one for each anchor)
        #    - Regression layer should output num_anchors * 4 channels (four coordinates for each anchor)
        self.cls_layer = nn.Conv2d(in_channels,
                                   self.num_anchors,
                                   kernel_size = 1,
                                   stride = 1)
        
        self.bbox_reg_layer = nn.Conv2d(in_channels,
                                    self.num_anchors * 4,
                                    kernel_size = 1,
                                    stride = 1)

        # DONE: Initialize the weights of the layers
        # - Normal distribution with std=0.01 for weights
        # - Constant 0 for biases
        for layer in [self.cls_layer, self.bbox_reg_layer]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def generate_anchors(self, image, feat):
        """Generate anchors for all feature map locations with all scales and aspect ratios"""
        # DONE: Implement anchor generation
        # 1. Get feature map dimensions and calculate stride relative to input image
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]

        stride_h = torch.tensor(image_h // grid_h, 
                            dtype = torch.int64,
                            device = feat.device)
        stride_w = torch.tensor(image_w // grid_w,
                            dtype = torch.int64,
                            device = feat.device)
        scales = torch.as_tensor(self.scales,
                                dtype = feat.dtype,
                                device = feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios,
                                    dtype = feat.dtype,
                                    device = feat.device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        
        # 2. Create base anchors at (0,0) for all combinations of scales and aspect ratios
        # base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = torch.stack([-ws / 2, -hs / 2, ws / 2, hs / 2], dim=1)
        base_anchors = base_anchors.round()

        # 3. Create shift values for all grid positions in the feature map
        shifts_x = torch.arange(0, grid_w,
                                dtype=torch.int32,
                                device = feat.device) * stride_w
        shifts_y = torch.arange(0, grid_h,
                                dtype=torch.int32,
                                device = feat.device) * stride_h
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x,
                                            indexing = "ij")
        
        shifts = torch.stack((shifts_x, 
                            shifts_y,
                            shifts_x,
                            shifts_y), dim=1)
        
        # 4. Combine base anchors with shifts to generate anchors for all positions
        anchors = (shifts.view(-1,1,4) + base_anchors.view(1,-1,4))
        # Return anchors with shape [num_locations*num_anchors, 4]
        anchors = anchors.reshape(-1, 4)
    
        return anchors  # Replace with your implementation
    
    def assign_targets_to_anchors(self, anchors, gt_boxes):
        """Assign ground truth boxes and labels to anchors based on IoU"""
        # DONE: Implement anchor-GT box matching
        # 1. Calculate IoU between all anchors and ground truth boxes
        iou_matrix = get_iou(gt_boxes, anchors)
        # 2. For each anchor, find the GT box with maximum IoU
        best_iou_per_anchor, best_gt_idx_per_anchor = iou_matrix.max(dim=0)  # shape (N,)
        best_gt_idx_before_threshold = best_gt_idx_per_anchor.clone()
        # 3. Label anchors based on IoU thresholds:
        #    - Positive (1): IoU > high_threshold
        #    - Negative (0): IoU < low_threshold
        #    - Ignore (-1): IoU between thresholds
        labels = torch.full((anchors.size(0),), -1, dtype=torch.int64, device=anchors.device)
        low_threshold = self.low_iou_threshold
        high_threshold = self.high_iou_threshold
        negative_mask = best_iou_per_anchor < low_threshold
        labels[negative_mask] = 0

        between_mask = (best_iou_per_anchor >= low_threshold) & (best_iou_per_anchor < high_threshold)
        labels[between_mask] = -1

        best_gt_idx_per_anchor[negative_mask] = -1
        best_gt_idx_per_anchor[between_mask] = -2

        
        # 4. For each GT box, find the anchor with highest IoU and label it positive
        best_iou_per_gt, _ = iou_matrix.max(dim=1)  # shape (M,)
        gt_anchor_pairs = torch.where(iou_matrix == best_iou_per_gt[:, None])
        
        forced_positive_anchors = gt_anchor_pairs[1]
        best_gt_idx_per_anchor[forced_positive_anchors] = best_gt_idx_before_threshold[forced_positive_anchors]

        matched_gt_boxes = gt_boxes[best_gt_idx_per_anchor.clamp(min=0)]
        # 5. Return labels and matched GT boxes for all anchors
        positive_mask = best_gt_idx_per_anchor >= 0
        labels[positive_mask] = 1

        labels = labels.to(dtype=torch.float32)

        return labels, matched_gt_boxes # Replace with your implementation

    def filter_proposals(self, proposals, cls_scores, image_shape):
        """Filter proposals using NMS and score thresholds"""
        # DONE: Implement proposal filtering
        # 1. Convert classification scores to probabilities
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)

        # 2. Select top-k proposals before NMS
        k_candidate = len(cls_scores)
        if k_candidate < 10000:
            k = k_candidate - 1
        else:
            k = 10000
        _, top_n_idx = cls_scores.topk(k)
        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]
        
        # 3. Clamp boxes to image boundary
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)

        # 4. Remove small boxes
        widths = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        keep = (widths >= 1) & (heights >= 1)
        proposals = proposals[keep]
        cls_scores = cls_scores[keep]

        # 5. Apply NMS
        keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals, cls_scores, 0.7)

        post_nms_keep_indices = keep_indices[
                                cls_scores[keep_indices].sort(descending=True)[1]
                                ]
        # 6. Take top-k after NMS
        proposals = proposals[post_nms_keep_indices[:2000]]
        cls_scores = cls_scores[post_nms_keep_indices[:2000]]
        
        # Return filtered proposals and their scores
        return proposals, cls_scores

    def forward(self, image, feat, target=None):
        """Forward pass for RPN"""
        # DONE: Implement RPN forward pass
        # 1. Apply convolutional layer
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        # 2. Generate classification and regression predictions
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)
        # 3. Generate anchors
        anchors = self.generate_anchors(image, feat)
        # 4. Reshape predictions to match anchor format
        number_of_anchors_per_location = cls_scores.size(1)
        cls_scores = cls_scores.permute(0,2,3,1)
        cls_scores = cls_scores.reshape(-1,1)

        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            number_of_anchors_per_location,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-1]
        )

        box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2)
        box_transform_pred = box_transform_pred.reshape(-1,4)
        
        # 5. Apply bounding box regression to anchors to get proposals
        proposals = apply_regression_pred_to_anchors_or_proposals(
            box_transform_pred.detach().reshape(-1,1,4),
            anchors
        )

        # 6. Filter proposals
        proposals = proposals.reshape(-1,4)
        proposals, scores = self.filter_proposals(proposals,
                                                  cls_scores.detach(),
                                                  image.shape)
        
        # Return a dictionary with proposals, scores, and (during training) losses
        rpn_output = {
            'proposals': proposals,
            'scores': scores
        }
        
        if not self.training or target is None:
            return rpn_output
        
        else:
            # During training (if target is provided):
            # 7. Assign targets to anchors
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                anchors, 
                target['bboxes'][0]
            )
            regression_targets = boxes_to_transformation_targets(
                matched_gt_boxes_for_anchors,
                anchors
            )
            # 8. Sample positive and negative anchors
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                labels_for_anchors,
                positive_count = 128,
                total_count = 256,
            )
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]

            # 9. Calculate classification and regression losses
            localization_loss = (
                torch.nn.functional.smooth_l1_loss(
                    box_transform_pred[sampled_pos_idx_mask],
                    regression_targets[sampled_pos_idx_mask],
                    beta = 1/9,
                    reduction = 'sum'
                ) / (sampled_idxs.numel())
            )
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_scores[sampled_idxs].flatten(),
                labels_for_anchors[sampled_idxs].flatten()
            )

            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_localization_loss'] = localization_loss

            return rpn_output # Replace with your implementation


# ======================================================================
# Part 3: RoI Feature Extraction and Part 4: Detection Head (40%)
# ======================================================================

class ROIHead(nn.Module):
    """ROI head for final classification and box refinement"""
    
    def __init__(self, model_config, num_classes, in_channels):
        super(ROIHead, self).__init__()
        # Store configuration
        self.num_classes = num_classes
        self.roi_batch_size = model_config['roi_batch_size']
        self.roi_pos_count = int(model_config['roi_pos_fraction'] * self.roi_batch_size)
        self.iou_threshold = model_config['roi_iou_threshold']
        self.low_bg_iou = model_config['roi_low_bg_iou']
        self.nms_threshold = model_config['roi_nms_threshold']
        self.topK_detections = model_config['roi_topk_detections']
        self.low_score_threshold = model_config['roi_score_threshold']
        self.pool_size = model_config['roi_pool_size']
        self.fc_inner_dim = model_config['fc_inner_dim']
        
        # DONE: Implement the network layers
        # 1. Two fully connected layers for feature transformation
        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size,
                             self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        # 2. Classification layer for predicting class scores
        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        # 3. Bounding box regression layer for refining boxes
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)

        # DONE: Initialize the weights
        # - Normal distribution with std=0.01 for classification layer
        # - Normal distribution with std=0.001 for box regression layer
        # - Constant 0 for biases
        nn.init.normal_(self.cls_layer.weight, std=0.01)
        nn.init.constant_(self.cls_layer.bias, 0)
        nn.init.normal_(self.bbox_reg_layer.weight, std=0.001)
        nn.init.constant_(self.bbox_reg_layer.bias, 0)

    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        """Assign ground truth boxes and labels to proposals based on IoU"""
        # DONE: Implement proposal-GT box matching
        # 1. Calculate IoU between proposals and ground truth boxes
        iou_matrix = get_iou(gt_boxes, proposals)
        # 2. For each proposal, find the GT box with maximum IoU
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        
        # 3. Label proposals based on IoU thresholds:
        #    - Background (0): IoU < iou_threshold
        #    - Ignore (-1): IoU < low_bg_iou
        #    - Class specific (1+): IoU >= iou_threshold
        below_low_threshold = best_match_iou < 0.5
        best_match_gt_idx[below_low_threshold] = -1
        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]

        labels = gt_labels[best_match_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)

        background_proposals = best_match_gt_idx == -1
        labels[background_proposals] = 0

         # 4. Return labels and matched GT boxes for all proposals
        return labels, matched_gt_boxes_for_proposals  # Replace with your implementation
    
    def forward(self, feat, proposals, image_shape, target):
        """Forward pass for ROI head"""
        # DONE: Implement ROI head forward pass
        if self.training and target is not None:
            gt_boxes = target['bboxes'][0]
            gt_labels = target['labels'][0]

            labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(
                proposals, gt_boxes, gt_labels
            )

            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                labels,
                positive_count = 32,
                total_count = 128
            )

            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)

            proposals = proposals[sampled_idxs]
            labels = labels[sampled_idxs]
            matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs]
            regression_targets = boxes_to_transformation_targets(
                matched_gt_boxes_for_proposals, proposals
            )
        
        spatial_scale = 0.0625

        proposal_roi_pool_feats = torchvision.ops.roi_pool(
            feat,
            [proposals],
            output_size = self.pool_size,
            spatial_scale = spatial_scale
        )

        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim = 1)
        box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))
        cls_scores = self.cls_layer(box_fc_7)
        box_transform_pred = self.bbox_reg_layer(box_fc_7)

        num_boxes, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)

        frcnn_output = {}
        # During training (if target is provided):
        # 1. Add ground truth boxes to proposals
        # 2. Assign targets to proposals
        # 3. Sample positive and negative proposals
        
        # For both training and inference:
        # 4. Calculate scale for RoI pooling
        # 5. Apply RoI pooling to extract features from proposals
        # 6. Apply fully connected layers
        # 7. Generate classification and box regression predictions
        
        # During training:
        # 8. Calculate classification and regression losses
        
        # During inference:
        # 9. Apply regression to proposals
        # 10. Filter and refine final predictions
        
        # Return a dictionary with boxes, scores, labels, and (during training) losses
        if self.training and target is not None:
            classification_loss = torch.nn.functional.cross_entropy(
                cls_scores,
                labels
            )
        
            fg_proposal_idxs = torch.where(labels > 0)[0]
            fg_class_labels = labels[fg_proposal_idxs]

            localization_loss = torch.nn.functional.smooth_l1_loss(
                box_transform_pred[fg_proposal_idxs, fg_class_labels],
                regression_targets[fg_proposal_idxs],
                beta = 1/9,
                reduction = 'sum'
            )

            localization_loss = localization_loss/labels.numel()
            frcnn_output['frcnn_classification_loss'] = classification_loss
            frcnn_output['frcnn_localization_loss'] = localization_loss
            return frcnn_output
        else:
            pred_boxes = apply_regression_pred_to_anchors_or_proposals(
                box_transform_pred,
                proposals
            )
            
            pred_scores = torch.nn.functional.softmax(cls_scores, dim = -1)

            pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape)

            pred_labels = torch.arange(num_classes, device=cls_scores.device)
            pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)

            pred_boxes = pred_boxes[:, 1:]
            pred_scores = pred_scores[:, 1:]
            pred_labels = pred_labels[:, 1:]

            pred_boxes = pred_boxes.reshape(-1, 4)
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_labels.reshape(-1)

            pred_boxes, pred_labels, pred_scores = self.filter_predictions(
                pred_boxes,
                pred_labels,
                pred_scores
            )

            pred_labels = pred_labels.to(torch.int64) #cast to int? - adjustment to pass visualization test

            frcnn_output['boxes'] = pred_boxes
            frcnn_output['scores'] = pred_scores
            frcnn_output['labels'] = pred_labels
            
            return frcnn_output
    
    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        """Filter predictions by score, size, and NMS"""
        # DONE: Implement prediction filtering
        # 1. Remove low scoring boxes
        # 2. Remove small boxes
        # 3. Apply per-class NMS
        # 4. Sort by score and take top-k
        # Return filtered boxes, labels, and scores
        keep = torch.where(pred_scores > 0.05)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        
        min_size = 1
        ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]

        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_indices = torch.where(pred_labels == class_id)[0]
            curr_keep_indices = torch.ops.torchvision.nms(
                pred_boxes[curr_indices],
                pred_scores[curr_indices],
                0.5
            )
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(
            descending = True
        )[1]]
        
        keep = post_nms_keep_indices[:100]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

        return pred_boxes, pred_scores, pred_labels  # Replace with your implementation


# ======================================================================
# Part 5: Faster R-CNN Model (20%)
# ======================================================================

class FasterRCNN(nn.Module):
    """Faster R-CNN object detection model"""
    
    def __init__(self, model_config, num_classes):
        super(FasterRCNN, self).__init__()
        self.model_config = model_config
        
        # VGG16 backbone (already implemented)
        vgg16 = torchvision.models.vgg16(weights="DEFAULT")
        self.backbone = vgg16.features[:-1]
        
        # DONE: Initialize the RPN and ROI head
        # 1. Create the RPN using model_config parameters
        # 2. Create the ROI head using model_config parameters
        self.rpn = RegionProposalNetwork(
            in_channels=512,
            scales = model_config['scales'],
            aspect_ratios = model_config['aspect_ratios'],
            model_config=model_config
            )
        self.roi_head = ROIHead(num_classes = num_classes,
                                in_channels = 512,
                                model_config = model_config)
        
        # Freeze early backbone layers (already implemented)
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        
        # Image normalization parameters
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']
    
    def normalize_resize_image_and_boxes(self, image, bboxes=None):
        """Normalize and resize image, adjusting bboxes accordingly"""
        # This method is already implemented
        # Handle batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Get original size
        c, h, w = image.shape[-3:]
        
        # Normalize image
        image = image.float()
        mean = torch.as_tensor(self.image_mean, dtype=image.dtype, device=image.device)
        std = torch.as_tensor(self.image_std, dtype=image.dtype, device=image.device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        
        # Calculate resize scale factor
        min_original_size = float(min((h, w)))
        max_original_size = float(max((h, w)))
        scale_factor_min = self.min_size / min_original_size
        
        # Adjust scale if needed to respect max size
        if max_original_size * scale_factor_min > self.max_size:
            scale_factor = self.max_size / max_original_size
        else:
            scale_factor = scale_factor_min
        
        # Resize image
        image = torch.nn.functional.interpolate(
            image, scale_factor=scale_factor, mode='bilinear',
            recompute_scale_factor=True, align_corners=False
        )
        
        # Resize bounding boxes if provided
        if bboxes is not None:
            # Handle different possible shapes of bboxes
            if bboxes.dim() == 2:  # Shape [num_boxes, 4]
                # Calculate resize ratios
                ratios = [
                    torch.tensor(s, dtype=torch.float32, device=bboxes.device) /
                    torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                    for s, s_orig in zip(image.shape[-2:], (h, w))
                ]
                ratio_height, ratio_width = ratios
                
                # Apply ratios to box coordinates
                xmin = bboxes[:, 0] * ratio_width
                ymin = bboxes[:, 1] * ratio_height
                xmax = bboxes[:, 2] * ratio_width
                ymax = bboxes[:, 3] * ratio_height
                
                bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
            
            elif bboxes.dim() == 3:  # Shape [batch_size, num_boxes, 4]
                # Calculate resize ratios
                ratios = [
                    torch.tensor(s, dtype=torch.float32, device=bboxes.device) /
                    torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                    for s, s_orig in zip(image.shape[-2:], (h, w))
                ]
                ratio_height, ratio_width = ratios
                
                # Apply ratios to box coordinates
                xmin, ymin, xmax, ymax = bboxes.unbind(2)
                xmin = xmin * ratio_width
                xmax = xmax * ratio_width
                ymin = ymin * ratio_height
                ymax = ymax * ratio_height
                
                bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
        
        return image, bboxes
    
    def forward(self, image, target=None):
        """Forward pass for Faster R-CNN"""
        # DONE: Implement the full Faster R-CNN forward pass
        # 1. Save original image shape
        # 2. Normalize and resize image (and boxes during training)
        # 3. Extract features with backbone
        # 4. Get region proposals from RPN
        # 5. Process proposals with ROI head
        # 6. During inference, transform boxes back to original image size
        # 7. Return RPN and FRCNN outputs
        old_shape = image.shape[-2:]
        if self.training:
            image, bboxes = self.normalize_resize_image_and_boxes(
                image, target['bboxes']
            )
            target['bboxes'] = bboxes
        else:
            image, _ = self.normalize_resize_image_and_boxes(
                image, None
            )

        feat = self.backbone(image)
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']

        frcnn_output = self.roi_head(feat, proposals, image.shape[-2:], target)
        
        if not self.training:
            frcnn_output['boxes'] = transform_boxes_to_original_size(
                frcnn_output['boxes'],
                image.shape[-2:],
                old_shape
            )

        return rpn_output, frcnn_output  # Replace with your implementation