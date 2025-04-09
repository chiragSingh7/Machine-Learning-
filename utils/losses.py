# utils/losses.py

import tensorflow as tf

# def get_anchor_targets(anchors, gt_boxes, pos_iou_thresh=0.7, neg_iou_thresh=0.3):
#     """
#     Generate positive and negative anchor indices for RPN training
    
#     Args:
#         anchors: [N, 4] tensor of anchor boxes (x1, y1, x2, y2)
#         gt_boxes: [G, 4] tensor of ground truth boxes (x1, y1, x2, y2)
#         pos_iou_thresh: IoU threshold for positive anchors
#         neg_iou_thresh: IoU threshold for negative anchors
        
#     Returns:
#         pos_indices: indices of positive anchors
#         neg_indices: indices of negative anchors
#     """
#     import tensorflow as tf
    
#     # Calculate IoU between each anchor and gt box
#     # First, expand dimensions to create [N, G, 4] tensors
#     anchors_exp = tf.expand_dims(anchors, axis=1)  # [N, 1, 4]
#     gt_boxes_exp = tf.expand_dims(gt_boxes, axis=0)  # [1, G, 4]
    
#     # Calculate intersection coordinates
#     x1 = tf.maximum(anchors_exp[..., 0], gt_boxes_exp[..., 0])
#     y1 = tf.maximum(anchors_exp[..., 1], gt_boxes_exp[..., 1])
#     x2 = tf.minimum(anchors_exp[..., 2], gt_boxes_exp[..., 2])
#     y2 = tf.minimum(anchors_exp[..., 3], gt_boxes_exp[..., 3])
    
#     # Calculate area of intersection
#     width = tf.maximum(0.0, x2 - x1)
#     height = tf.maximum(0.0, y2 - y1)
#     intersection = width * height
    
#     # Calculate area of anchors and gt boxes
#     area_anchors = (anchors[..., 2] - anchors[..., 0]) * (anchors[..., 3] - anchors[..., 1])
#     area_gt = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])
    
#     # Expand dimensions for broadcasting
#     area_anchors = tf.expand_dims(area_anchors, axis=1)  # [N, 1]
#     area_gt = tf.expand_dims(area_gt, axis=0)  # [1, G]
    
#     # Calculate union
#     union = area_anchors + area_gt - intersection
    
#     # Calculate IoU
#     iou = intersection / union  # [N, G]
    
#     # Get max IoU for each anchor
#     max_iou = tf.reduce_max(iou, axis=1)  # [N]
    
#     # Generate positive and negative indices
#     pos_indices = tf.where(max_iou >= pos_iou_thresh)[:, 0]
#     neg_indices = tf.where(max_iou < neg_iou_thresh)[:, 0]

#     pos_indices = tf.cast(pos_indices, tf.int32)
#     neg_indices = tf.cast(neg_indices, tf.int32)
    
#     # Handle empty tensors case
#     if tf.size(pos_indices) == 0:
#         pos_indices = tf.zeros([0], dtype=tf.int32)
#     if tf.size(neg_indices) == 0:
    #     neg_indices = tf.zeros([0], dtype=tf.int32)
    
    # return pos_indices, neg_indices


def get_anchor_targets(anchors, gt_boxes, pos_iou_thresh=0.7, neg_iou_thresh=0.3):
    """Use NumPy operations for more reliable behavior in tf.py_function"""
    import numpy as np
    
    # Convert TF tensors to NumPy arrays
    anchors_np = anchors.numpy()
    gt_boxes_np = gt_boxes.numpy()
    
    if gt_boxes_np.shape[0] == 0:  # No ground truth boxes
        pos_indices = np.array([], dtype=np.int32)
        neg_indices = np.arange(len(anchors_np), dtype=np.int32)
        return pos_indices, neg_indices
    
    # Calculate IoUs
    ious = np.zeros((len(anchors_np), len(gt_boxes_np)))
    
    for i, anchor in enumerate(anchors_np):
        for j, gt in enumerate(gt_boxes_np):
            # Calculate intersection area
            x1 = max(anchor[0], gt[0])
            y1 = max(anchor[1], gt[1])
            x2 = min(anchor[2], gt[2])
            y2 = min(anchor[3], gt[3])
            
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            intersection = w * h
            
            # Calculate areas
            anchor_area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
            gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
            
            # Calculate IoU
            union = anchor_area + gt_area - intersection
            ious[i, j] = intersection / union if union > 0 else 0
    
    # Get max IoU for each anchor
    max_ious = np.max(ious, axis=1)
    
    # Get positive and negative indices
    pos_indices = np.where(max_ious >= pos_iou_thresh)[0].astype(np.int32)
    neg_indices = np.where(max_ious < neg_iou_thresh)[0].astype(np.int32)
    
    return pos_indices, neg_indices


def compute_iou(boxes1, boxes2):
    """
    Compute pairwise IoU between two sets of boxes.
    boxes1: [N,4], boxes2: [M,4] in (x1,y1,x2,y2)
    Returns: [N,M] IoU matrix
    (See Eqn (8): IoU = area_inter / area_union)
    """
    b1 = tf.expand_dims(boxes1, 1)  # [N,1,4]
    b2 = tf.expand_dims(boxes2, 0)  # [1,M,4]
    x1 = tf.maximum(b1[...,0], b2[...,0])
    y1 = tf.maximum(b1[...,1], b2[...,1])
    x2 = tf.minimum(b1[...,2], b2[...,2])
    y2 = tf.minimum(b1[...,3], b2[...,3])
    inter_w = tf.maximum(0.0, x2 - x1)
    inter_h = tf.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
    area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
    union = tf.expand_dims(area1,1) + tf.expand_dims(area2,0) - inter
    return inter / (union + 1e-8)

def encode_boxes(anchors, gt_boxes):
    """
    Encode ground-truth boxes with respect to anchors.
    anchors, gt_boxes: [N,4] (x1,y1,x2,y2)
    Returns: [N,4] deltas (dx,dy,dw,dh)
    (Used in RPN & RCNN regression losses; cf. Eqn (6))
    """
    xa = (anchors[:,0] + anchors[:,2]) * 0.5
    ya = (anchors[:,1] + anchors[:,3]) * 0.5
    wa = anchors[:,2] - anchors[:,0]
    ha = anchors[:,3] - anchors[:,1]

    x = (gt_boxes[:,0] + gt_boxes[:,2]) * 0.5
    y = (gt_boxes[:,1] + gt_boxes[:,3]) * 0.5
    w =  gt_boxes[:,2] - gt_boxes[:,0]
    h =  gt_boxes[:,3] - gt_boxes[:,1]

    dx = (x - xa) / (wa + 1e-8)
    dy = (y - ya) / (ha + 1e-8)
    dw = tf.math.log(w / (wa + 1e-8))
    dh = tf.math.log(h / (ha + 1e-8))

    return tf.stack([dx, dy, dw, dh], axis=1)

def smooth_l1_loss(diff):
    """
    Smooth L1 loss elementwise:
    0.5 * diff^2        if |diff| < 1
    |diff| - 0.5        otherwise
    (Eqn (6))
    """
    abs_diff = tf.abs(diff)
    sq = 0.5 * diff * diff
    return tf.where(abs_diff < 1.0, sq, abs_diff - 0.5)

def rpn_class_loss(rpn_class_logits, anchors, gt_boxes):
    """RPN classification loss"""
    # Reshape tensors if needed
    B = tf.shape(rpn_class_logits)[0]
    
    total_loss = 0.0
    for b in range(B):
        # Get this batch's data
        logits_b = rpn_class_logits[b]  # [N, 2]
        anchors_b = anchors  # [N, 4] - anchors are the same for all batches
        gt_boxes_b = gt_boxes[b]  # [G, 4]
        
        # Get indices for positive and negative examples
        pos_indices, neg_indices = tf.py_function(
            func=lambda a, g: get_anchor_targets(a, g),
            inp=[anchors_b, gt_boxes_b],
            Tout=[tf.int32, tf.int32]
        )
        
        # Set shapes that get lost in py_function
        pos_indices.set_shape([None])
        neg_indices.set_shape([None])
        
        # Handle case where indices might be empty
        if tf.size(pos_indices) == 0 and tf.size(neg_indices) == 0:
            # No positive or negative samples - return zero loss
            continue
        
        # Positive loss
        pos_loss = 0.0
        if tf.size(pos_indices) > 0:
            pos_logits = tf.gather(logits_b, pos_indices)
            pos_labels = tf.ones([tf.shape(pos_indices)[0]], dtype=tf.int32)
            pos_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=pos_labels, logits=pos_logits
                )
            )
        
        # Negative loss
        neg_loss = 0.0
        if tf.size(neg_indices) > 0:
            neg_logits = tf.gather(logits_b, neg_indices)
            neg_labels = tf.zeros([tf.shape(neg_indices)[0]], dtype=tf.int32)
            neg_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=neg_labels, logits=neg_logits
                )
            )
        
        # Calculate batch loss
        if tf.size(pos_indices) > 0 and tf.size(neg_indices) > 0:
            batch_loss = (pos_loss + neg_loss) / 2.0
        elif tf.size(pos_indices) > 0:
            batch_loss = pos_loss
        else:
            batch_loss = neg_loss
            
        total_loss += batch_loss
    
    return total_loss / tf.cast(B, tf.float32)

def rpn_bbox_loss(rpn_bbox_preds, anchors, gt_boxes,
                  pos_iou_thresh=0.7):
    """
    RPN bounding-box regression loss (Smooth L1).
    rpn_bbox_preds: [B,N,4] predicted deltas
    anchors:        [N,4]
    gt_boxes:       [B,M,4]
    (Only positive anchors; Eqn (6))
    """
    B = tf.shape(rpn_bbox_preds)[0]
    losses = []
    for b in range(B):
        preds = rpn_bbox_preds[b]  # [N,4]
        boxes = gt_boxes[b]        # [M,4]
        if tf.shape(boxes)[0] == 0:
            losses.append(0.0)
            continue
        ious = compute_iou(anchors, boxes)    # [N,M]
        max_iou = tf.reduce_max(ious, axis=1)
        pos = tf.where(max_iou >= pos_iou_thresh)[:,0]
        if tf.size(pos) == 0:
            losses.append(0.0)
            continue
        # match each anchor to best GT
        gt_idx = tf.argmax(ious, axis=1)
        gt_for_anchors = tf.gather(boxes, gt_idx)     # [N,4]
        true_deltas = encode_boxes(anchors, gt_for_anchors)
        preds_pos   = tf.gather(preds, pos)
        true_pos    = tf.gather(true_deltas, pos)
        diff = preds_pos - true_pos
        l1   = smooth_l1_loss(diff)
        losses.append(tf.reduce_mean(tf.reduce_sum(l1, axis=1)))
    return tf.add_n(losses) / tf.cast(B, tf.float32)

def rcnn_class_loss(rcnn_class_logits, proposals, gt_boxes, pos_iou_thresh=0.5):
    """
    RCNN classification loss (multi-class crossentropy).
    rcnn_class_logits: [B*P, C] raw logits
    proposals:         [B,P,4]
    gt_boxes:          [B,M,4]
    (Match proposals→gt by IoU; Eqn (7))
    """
    B = tf.shape(proposals)[0]
    P = tf.shape(proposals)[1]
    losses = []
    for b in range(B):
        start = b * P
        end   = start + P
        logits = rcnn_class_logits[start:end]  # [P,C]
        props  = proposals[b]                  # [P,4]
        boxes  = gt_boxes[b]                   # [M,4]
        N = tf.shape(props)[0]
        labels = -1 * tf.ones((N,), dtype=tf.int32)
        if tf.shape(boxes)[0] > 0:
            ious = compute_iou(props, boxes)
            max_iou = tf.reduce_max(ious, axis=1)
            
            # Cast the indices to int32 here
            pos = tf.cast(tf.where(max_iou >= pos_iou_thresh)[:,0], tf.int32)
            neg = tf.cast(tf.where(max_iou < pos_iou_thresh)[:,0], tf.int32)
            
            labels = tf.tensor_scatter_nd_update(labels,
                                                 tf.expand_dims(pos,1),
                                                 tf.ones_like(pos))
            labels = tf.tensor_scatter_nd_update(labels,
                                                 tf.expand_dims(neg,1),
                                                 tf.zeros_like(neg))
        else:
            labels = tf.zeros((N,), dtype=tf.int32)
            
        # Cast this to int32 as well
        valid = tf.cast(tf.where(labels >= 0)[:,0], tf.int32)
        
        lbl   = tf.gather(labels, valid)
        lg    = tf.gather(logits, valid)
        ce    = tf.keras.losses.sparse_categorical_crossentropy(
                    lbl, lg, from_logits=True)
        losses.append(tf.reduce_mean(ce))
    return tf.add_n(losses) / tf.cast(B, tf.float32)

def rcnn_bbox_loss(rcnn_bbox_preds, proposals, gt_boxes,
                   pos_iou_thresh=0.5):
    """
    RCNN bounding-box regression loss (Smooth L1).
    rcnn_bbox_preds: [B*P, C*4]
    proposals:       [B,P,4]
    gt_boxes:        [B,M,4]
    (Only positive proposals; Eqn (6))
    """
    B = tf.shape(proposals)[0]
    P = tf.shape(proposals)[1]
    losses = []
    for b in range(B):
        start = b * P
        end   = start + P
        preds = rcnn_bbox_preds[start:end]  # [P,C*4]
        props = proposals[b]                # [P,4]
        boxes = gt_boxes[b]                 # [M,4]
        if tf.shape(boxes)[0] == 0:
            losses.append(0.0)
            continue
        ious = compute_iou(props, boxes)
        max_iou = tf.reduce_max(ious, axis=1)
        pos = tf.where(max_iou >= pos_iou_thresh)[:,0]
        if tf.size(pos) == 0:
            losses.append(0.0)
            continue
        gt_idx = tf.argmax(ious, axis=1)
        gt_for_props = tf.gather(boxes, gt_idx)
        # encode and gather only class-specific deltas
        true_deltas = encode_boxes(props, gt_for_props)  # [P,4]
        # assuming class-agnostic: use preds[:, :4]
        pred_deltas = preds[:, :4]
        pred_pos = tf.gather(pred_deltas, pos)
        true_pos= tf.gather(true_deltas, pos)
        diff = pred_pos - true_pos
        l1   = smooth_l1_loss(diff)
        losses.append(tf.reduce_mean(tf.reduce_sum(l1, axis=1)))
    return tf.add_n(losses) / tf.cast(B, tf.float32)

def mask_loss(mask_preds, gt_masks, proposals, gt_boxes,
              pos_iou_thresh=0.5):
    """
    Mask head loss (binary crossentropy).
    mask_preds: [B*P, Hm, Wm, C]
    gt_masks:   [B,H,W,1]
    proposals:  [B,P,4]
    gt_boxes:   [B,M,4]
    (Only positive proposals; Eqn (9) flatten & Eqn (10) FC not used here)
    """
    B = tf.shape(gt_masks)[0]
    P = tf.shape(proposals)[1]
    losses = []
    for b in range(B):
        props = proposals[b]
        boxes = gt_boxes[b]
        if tf.shape(boxes)[0] == 0:
            continue
        ious = compute_iou(props, boxes)
        max_iou = tf.reduce_max(ious, axis=1)
        pos = tf.where(max_iou >= pos_iou_thresh)[:,0]
        if tf.size(pos) == 0:
            continue
        # normalize & crop gt mask for each positive ROI
        H = tf.shape(gt_masks)[1]
        W = tf.shape(gt_masks)[2]
        boxes_pos = tf.gather(props, pos)
        y1 = boxes_pos[:,1] / tf.cast(H, tf.float32)
        x1 = boxes_pos[:,0] / tf.cast(W, tf.float32)
        y2 = boxes_pos[:,3] / tf.cast(H, tf.float32)
        x2 = boxes_pos[:,2] / tf.cast(W, tf.float32)
        boxes_norm = tf.stack([y1,x1,y2,x2], axis=1)
        box_inds = tf.zeros_like(pos) + b
        gt_crop = tf.image.crop_and_resize(
            gt_masks, boxes_norm, box_inds,
            crop_size=(tf.shape(mask_preds)[1], tf.shape(mask_preds)[2]),
            method='nearest'
        )[...,0]
        # gather predicted masks for fg class (index 1)
        start = b * P
        idxs  = start + pos
        pred = tf.gather(mask_preds, idxs)[...,1]
        bce  = tf.keras.losses.binary_crossentropy(gt_crop, pred)
        losses.append(tf.reduce_mean(bce))
    if losses:
        return tf.add_n(losses) / tf.cast(len(losses), tf.float32)
    else:
        return tf.constant(0.0)
