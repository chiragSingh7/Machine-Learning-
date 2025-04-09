import tensorflow as tf
from model.rpn import RegionProposalNetwork
from model.roi_head import RoIHead

class FasterRCNN(tf.keras.Model):
    def __init__(self, backbone, num_classes=2, image_shape=(480,854)):
        super().__init__()
        self.backbone = backbone
        self.rpn      = RegionProposalNetwork(image_shape=image_shape)
        self.roi_head = RoIHead(num_classes, image_shape=image_shape)

    def call(self, images, gt_boxes=None, gt_masks=None, training=False):
        features = self.backbone(images, training=training)
        proposals, rpn_losses = self.rpn(features, gt_boxes=gt_boxes)
        detections, rcnn_losses = self.roi_head(
            features, proposals,
            gt_boxes=gt_boxes, gt_masks=gt_masks
        )
        detections['proposals'] = proposals
        losses = {**rpn_losses, **rcnn_losses}
        return detections, losses
