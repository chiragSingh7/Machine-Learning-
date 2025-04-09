import tensorflow as tf
from model.rpn import RegionProposalNetwork
from model.roi_head import RoIHead

class FasterRCNN(tf.keras.Model):
    def __init__(self,
                 backbone,
                 num_classes=2,
                 image_shape=(384,640),
                 score_thresh=0.5,
                 iou_thresh=0.5,
                 max_detections=100):
        super().__init__()
        self.backbone       = backbone
        self.rpn            = RegionProposalNetwork(image_shape=image_shape)
        self.roi_head       = RoIHead(num_classes, image_shape=image_shape)
        self.score_thresh   = score_thresh
        self.iou_thresh     = iou_thresh
        self.max_detections = max_detections

    def call(self, images, gt_boxes=None, gt_masks=None, training=False):
        # 1) Backbone
        features = self.backbone(images, training=training)

        # 2) RPN
        proposals, rpn_losses = self.rpn(features, gt_boxes=gt_boxes)

        # 3) ROI Head
        dets, rcnn_losses = self.roi_head(
            features, proposals,
            gt_boxes=gt_boxes, gt_masks=gt_masks
        )
        dets['proposals'] = proposals

        # 4) Inference post‐processing
        if not training:
            boxes, scores, classes, masks = self._postprocess(
                dets['proposals'][0],
                dets['class_logits'],
                dets['bbox_preds'],
                dets['mask_preds']
            )
            return {
                'boxes':   boxes,
                'scores':  scores,
                'classes': classes,
                'masks':   masks
            }, {}

        # 5) Training returns raw dets + losses
        losses = {**rpn_losses, **rcnn_losses}
        return dets, losses

    def _postprocess(self, proposals, cls_logits, bbox_preds, mask_preds):
        probs  = tf.nn.softmax(cls_logits, axis=-1)
        scores = probs[:,1]
        deltas = bbox_preds[:,4:8]  # class=1
        boxes  = self.rpn._decode_boxes(proposals, deltas)

        H, W = self.rpn.image_shape
        x1 = tf.clip_by_value(boxes[:,0], 0, W-1)
        y1 = tf.clip_by_value(boxes[:,1], 0, H-1)
        x2 = tf.clip_by_value(boxes[:,2], 0, W-1)
        y2 = tf.clip_by_value(boxes[:,3], 0, H-1)
        boxes = tf.stack([x1,y1,x2,y2], axis=1)

        keep = tf.image.non_max_suppression(
            boxes, scores,
            max_output_size=self.max_detections,
            iou_threshold=self.iou_thresh,
            score_threshold=self.score_thresh
        )
        boxes   = tf.gather(boxes, keep)
        scores  = tf.gather(scores, keep)
        classes = tf.ones_like(scores, dtype=tf.int32)
        masks   = tf.gather(mask_preds[:,:,:,1], keep)
        return boxes, scores, classes, masks
