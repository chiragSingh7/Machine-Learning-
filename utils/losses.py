import tensorflow as tf

def _smooth_l1(x):
    abs_x = tf.abs(x)
    return tf.where(abs_x<1.0, 0.5*tf.square(abs_x), abs_x-0.5)

def _bbox_transform(anchors, gt):
    wa = anchors[:,2]-anchors[:,0]; ha = anchors[:,3]-anchors[:,1]
    xa = anchors[:,0]+0.5*wa; ya = anchors[:,1]+0.5*ha
    wg = gt[:,2]-gt[:,0]; hg = gt[:,3]-gt[:,1]
    xg = gt[:,0]+0.5*wg; yg = gt[:,1]+0.5*hg
    tx = (xg-xa)/wa; ty = (yg-ya)/ha
    tw = tf.math.log(wg/wa); th = tf.math.log(hg/ha)
    return tf.stack([tx,ty,tw,th],axis=1)

def _compute_iou(boxes1, boxes2):
    x11,y11,x12,y12 = tf.split(boxes1,4,axis=1)
    x21,y21,x22,y22 = tf.split(boxes2,4,axis=1)
    x11 = tf.expand_dims(x11,1); y11 = tf.expand_dims(y11,1)
    x12 = tf.expand_dims(x12,1); y12 = tf.expand_dims(y12,1)
    inter_x1 = tf.maximum(x11,x21); inter_y1 = tf.maximum(y11,y21)
    inter_x2 = tf.minimum(x12,x22); inter_y2 = tf.minimum(y12,y22)
    inter_w = tf.maximum(0.0, inter_x2-inter_x1)
    inter_h = tf.maximum(0.0, inter_y2-inter_y1)
    inter   = inter_w * inter_h
    area1   = (x12-x11)*(y12-y11)
    area2   = (x22-x21)*(y22-y21)
    union   = area1 + area2 - inter
    return inter/(union+1e-7)

def rpn_class_loss(cls_logits, anchors, gt_boxes,
                   pos_thresh=0.7, neg_thresh=0.3):
    B = tf.shape(cls_logits)[0]
    losses=[]
    for b in range(B):
        logits = cls_logits[b]  # [N,2]
        gtb    = gt_boxes[b]    # [M,4]
        ious   = _compute_iou(anchors, gtb)
        max_i  = tf.reduce_max(ious,1)
        max_i  = tf.reshape(max_i,[-1])
        pos    = max_i>=pos_thresh
        neg    = max_i< neg_thresh
        valid  = tf.logical_or(pos,neg)
        labels = tf.cast(pos,tf.int32)
        lv = tf.boolean_mask(labels, valid)
        lo = tf.boolean_mask(logits, valid)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=lv, logits=lo)
        losses.append(tf.reduce_mean(loss))
    return tf.add_n(losses)/tf.cast(B,tf.float32)

def rpn_bbox_loss(reg_preds, anchors, gt_boxes,
                  pos_thresh=0.7):
    B = tf.shape(reg_preds)[0]
    losses=[]
    for b in range(B):
        preds = reg_preds[b]  # [N,4]
        gtb   = gt_boxes[b]   # [M,4]
        ious  = _compute_iou(anchors, gtb)
        max_i = tf.reduce_max(ious,1)
        max_i = tf.reshape(max_i,[-1])
        pos   = max_i>=pos_thresh
        num_pos = tf.reduce_sum(tf.cast(pos,tf.float32))
        if tf.equal(num_pos,0.0):
            losses.append(0.0); continue
        idxs    = tf.argmax(ious,1)
        matched = tf.gather(gtb,idxs)
        targets = _bbox_transform(anchors, matched)
        pp = tf.boolean_mask(preds, pos)
        tg = tf.boolean_mask(targets,pos)
        diffs= pp - tg
        loss = _smooth_l1(diffs)
        losses.append(tf.reduce_mean(tf.reduce_sum(loss,1)))
    return tf.add_n(losses)/tf.cast(B,tf.float32)

def rcnn_class_loss(cls_logits, proposals, gt_boxes,
                    pos_thresh=0.5, neg_thresh=0.5):
    B = tf.shape(proposals)[0]
    P = tf.shape(proposals)[1]
    C = tf.shape(cls_logits)[1]
    logits = tf.reshape(cls_logits,[B,P,C])
    losses=[]
    for b in range(B):
        logb = logits[b]  # [P,C]
        prop = proposals[b]
        gtb  = gt_boxes[b]
        ious = _compute_iou(prop, gtb)
        max_i= tf.reduce_max(ious,1)
        max_i= tf.reshape(max_i,[-1])
        pos  = max_i>=pos_thresh
        neg  = max_i< neg_thresh
        valid= tf.logical_or(pos,neg)
        labels = tf.cast(pos,tf.int32)
        lv = tf.boolean_mask(labels, valid)
        lo = tf.boolean_mask(logb, valid)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=lv, logits=lo)
        losses.append(tf.reduce_mean(loss))
    return tf.add_n(losses)/tf.cast(B,tf.float32)

def rcnn_bbox_loss(bbox_preds, proposals, gt_boxes,
                   pos_thresh=0.5):
    B = tf.shape(proposals)[0]
    P = tf.shape(proposals)[1]
    C4= tf.shape(bbox_preds)[1]//4
    preds = tf.reshape(bbox_preds,[B,P,C4,4])
    losses=[]
    for b in range(B):
        dp = preds[b,: ,1,:]  # [P,4]
        prop= proposals[b]
        gtb = gt_boxes[b]
        ious= _compute_iou(prop, gtb)
        max_i= tf.reduce_max(ious,1)
        max_i= tf.reshape(max_i,[-1])
        pos  = max_i>=pos_thresh
        num_pos = tf.reduce_sum(tf.cast(pos,tf.float32))
        if tf.equal(num_pos,0.0):
            losses.append(0.0); continue
        idxs = tf.argmax(ious,1)
        matched = tf.gather(gtb,idxs)
        targets = _bbox_transform(prop, matched)
        dp_p = tf.boolean_mask(dp, pos)
        tg_p = tf.boolean_mask(targets,pos)
        diffs= dp_p - tg_p
        loss = _smooth_l1(diffs)
        losses.append(tf.reduce_mean(tf.reduce_sum(loss,1)))
    return tf.add_n(losses)/tf.cast(B,tf.float32)

def mask_loss(mask_preds, gt_masks, proposals, gt_boxes,
              mask_size=28, pos_thresh=0.5):
    B = tf.shape(proposals)[0]
    P = tf.shape(proposals)[1]
    ms = mask_size

    preds = tf.reshape(mask_preds,[B,P,ms,ms,-1])
    losses=[]
    for b in range(B):
        mp   = preds[b,:,:,:,1]  # [P,ms,ms]
        prop = proposals[b]
        gtm  = gt_masks[b:b+1]    # [1,H,W,1]
        gtb  = gt_boxes[b]
        ious = _compute_iou(prop, gtb)
        max_i= tf.reduce_max(ious,1)
        max_i= tf.reshape(max_i,[-1])
        pos_idxs = tf.where(max_i>=pos_thresh)[:,0]
        K = tf.shape(pos_idxs)[0]
        if tf.equal(K,0):
            losses.append(0.0); continue

        H = tf.cast(tf.shape(gtm)[1],tf.float32)
        W = tf.cast(tf.shape(gtm)[2],tf.float32)
        boxes = tf.gather(prop,pos_idxs)
        y1 = boxes[:,1]/H; x1 = boxes[:,0]/W
        y2 = boxes[:,3]/H; x2 = boxes[:,2]/W
        bxs = tf.stack([y1,x1,y2,x2],axis=1)
        idxs = tf.zeros([K],dtype=tf.int32)

        crops = tf.image.crop_and_resize(
            tf.cast(gtm,tf.float32), bxs, idxs,
            crop_size=(ms,ms), method='nearest'
        )[...,0]  # [K,ms,ms]

        pm = tf.gather(mp,pos_idxs)
        bce = tf.keras.losses.binary_crossentropy(
            crops, pm, from_logits=False
        )
        losses.append(tf.reduce_mean(bce))

    return tf.add_n(losses)/tf.cast(B,tf.float32)
