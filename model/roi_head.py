# model/roi_head.py
import tensorflow as tf
from utils.losses import rcnn_class_loss, rcnn_bbox_loss, mask_loss

class RoIHead(tf.keras.layers.Layer):
    def __init__(self, num_classes, pool_size=7, mask_size=28,
                 image_shape=(480,854), **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.pool_size    = pool_size
        self.mask_size    = mask_size
        self.image_shape  = image_shape  # (H,W)

        # detection head
        self.flatten      = tf.keras.layers.Flatten()
        self.fc1          = tf.keras.layers.Dense(1024, activation='relu')
        self.fc2          = tf.keras.layers.Dense(1024, activation='relu')
        self.class_logits = tf.keras.layers.Dense(num_classes)
        self.bbox_preds   = tf.keras.layers.Dense(num_classes*4)

        # mask head
        self.conv1     = tf.keras.layers.Conv2D(256,3,padding='same',activation='relu')
        self.conv2     = tf.keras.layers.Conv2D(256,3,padding='same',activation='relu')
        self.deconv    = tf.keras.layers.Conv2DTranspose(256,2,strides=2,activation='relu')
        self.mask_pred = tf.keras.layers.Conv2D(num_classes,1,activation='sigmoid')

    def call(self, feature_map, proposals, gt_boxes=None, gt_masks=None):
        """
        feature_map: [B,Hf,Wf,C]
        proposals:   [B,P,4] in x1,y1,x2,y2
        """
        B = tf.shape(feature_map)[0]
        P = tf.shape(proposals)[1]
        H, W = self.image_shape

        # normalize proposals to [y1,x1,y2,x2] in [0,1]
        y1 = proposals[:,:,1] / tf.cast(H, tf.float32)
        x1 = proposals[:,:,0] / tf.cast(W, tf.float32)
        y2 = proposals[:,:,3] / tf.cast(H, tf.float32)
        x2 = proposals[:,:,2] / tf.cast(W, tf.float32)
        boxes = tf.stack([y1,x1,y2,x2], axis=2)  # [B,P,4]

        # flatten for crop_and_resize
        box_inds   = tf.repeat(tf.range(B), P)
        boxes_flat = tf.reshape(boxes, [-1,4])

        pooled = tf.image.crop_and_resize(
            feature_map, boxes_flat, box_inds,
            crop_size=(self.pool_size, self.pool_size),
            method='bilinear'
        )  # [B*P, pool_size, pool_size, C]

        # detection branch
        x = self.flatten(pooled)
        x = self.fc1(x); x = self.fc2(x)
        cls_logits = self.class_logits(x)  # [B*P, num_classes]
        bbox_preds = self.bbox_preds(x)    # [B*P, num_classes*4]

        # mask branch
        m = pooled
        m = self.conv1(m); m = self.conv2(m)
        m = self.deconv(m)
        mask_preds = self.mask_pred(m)     # [B*P, mask_size, mask_size, num_classes]

        detections = {
            'class_logits': cls_logits,
            'bbox_preds':   bbox_preds,
            'mask_preds':   mask_preds
        }

        losses = {}
        if gt_boxes is not None and gt_masks is not None:
            losses['rcnn_class_loss'] = rcnn_class_loss(
                cls_logits, proposals, gt_boxes
            )
            losses['rcnn_bbox_loss']  = rcnn_bbox_loss(
                bbox_preds, proposals, gt_boxes
            )
            losses['mask_loss']       = mask_loss(
                mask_preds, gt_masks, proposals, gt_boxes
            )
        return detections, losses
