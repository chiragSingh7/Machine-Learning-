import tensorflow as tf
from utils.losses import rpn_class_loss, rpn_bbox_loss

class RegionProposalNetwork(tf.keras.layers.Layer):
    def __init__(self,
                 anchor_scales=[8,16,32],
                 anchor_ratios=[0.5,1.0,2.0],
                 pre_nms_top_n=6000,
                 post_nms_top_n=300,
                 nms_thresh=0.7,
                 image_shape=(384,640),
                 **kwargs):
        super().__init__(**kwargs)
        self.anchor_scales  = anchor_scales
        self.anchor_ratios  = anchor_ratios
        self.num_anchors    = len(anchor_scales)*len(anchor_ratios)
        self.pre_nms_top_n  = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh     = nms_thresh
        self.image_shape    = image_shape

        self.conv_shared = tf.keras.layers.Conv2D(512,3,padding='same',
                                                  activation='relu')
        self.conv_cls    = tf.keras.layers.Conv2D(self.num_anchors*2,1)
        self.conv_reg    = tf.keras.layers.Conv2D(self.num_anchors*4,1)

    def call(self, feature_map, gt_boxes=None):
        B  = tf.shape(feature_map)[0]
        Hf = tf.shape(feature_map)[1]
        Wf = tf.shape(feature_map)[2]

        x = self.conv_shared(feature_map)
        cls_logits = self.conv_cls(x)   # [B,Hf,Wf,A*2]
        bbox_preds = self.conv_reg(x)   # [B,Hf,Wf,A*4]

        proposals = self._generate_proposals(cls_logits, bbox_preds)
        losses    = {}
        if gt_boxes is not None:
            cls_flat = tf.reshape(cls_logits,[B,-1,2])
            reg_flat = tf.reshape(bbox_preds,[B,-1,4])
            anchors  = self._generate_anchors(Hf,Wf)
            anchors_flat = tf.reshape(anchors,[-1,4])
            losses['rpn_class_loss'] = rpn_class_loss(
                cls_flat, anchors_flat, gt_boxes)
            losses['rpn_bbox_loss']  = rpn_bbox_loss(
                reg_flat, anchors_flat, gt_boxes)
        return proposals, losses

    def _generate_anchors(self, Hf, Wf):
        stride = 32.0
        y_centers = (tf.cast(tf.range(Hf),tf.float32)+0.5)*stride
        x_centers = (tf.cast(tf.range(Wf),tf.float32)+0.5)*stride
        y_centers = tf.reshape(y_centers,[Hf,1])
        x_centers = tf.reshape(x_centers,[1,Wf])
        y_centers = tf.tile(y_centers,[1,Wf])
        x_centers = tf.tile(x_centers,[Hf,1])

        ws, hs = [], []
        for s in self.anchor_scales:
            for r in self.anchor_ratios:
                hs.append(s*tf.sqrt(r))
                ws.append(s/tf.sqrt(r))
        ws = tf.stack(ws); hs = tf.stack(hs)

        K = Hf*Wf
        yc = tf.reshape(y_centers,[K,1])
        xc = tf.reshape(x_centers,[K,1])
        w  = tf.reshape(ws,[1,-1])
        h  = tf.reshape(hs,[1,-1])

        x1 = xc - w/2; y1 = yc - h/2
        x2 = xc + w/2; y2 = yc + h/2
        anchors = tf.stack([x1,y1,x2,y2],axis=2)  # [K,A,4]
        anchors = tf.reshape(anchors,[Hf,Wf,self.num_anchors,4])
        return anchors

    def _decode_boxes(self, anchors, deltas):
        xa = (anchors[:,0]+anchors[:,2])*0.5
        ya = (anchors[:,1]+anchors[:,3])*0.5
        wa = anchors[:,2]-anchors[:,0]
        ha = anchors[:,3]-anchors[:,1]
        dx,dy,dw,dh = tf.split(deltas,4,axis=1)
        x = dx*wa + xa; y = dy*ha + ya
        w = tf.exp(dw)*wa; h = tf.exp(dh)*ha
        x1 = x-w/2; y1 = y-h/2
        x2 = x+w/2; y2 = y+h/2
        return tf.concat([x1,y1,x2,y2],axis=1)

    def _generate_proposals(self, cls_logits, bbox_preds):
        B  = tf.shape(cls_logits)[0]
        Hf = tf.shape(cls_logits)[1]
        Wf = tf.shape(cls_logits)[2]

        scores = tf.nn.softmax(
            tf.reshape(cls_logits,[B,-1,2]),axis=-1)[...,1]
        deltas = tf.reshape(bbox_preds,[B,-1,4])
        anchors = tf.reshape(self._generate_anchors(Hf,Wf),[-1,4])

        all_props = []
        for b in range(B):
            sc = scores[b]; dt = deltas[b]
            props = self._decode_boxes(anchors, dt)
            H,W = self.image_shape
            x1 = tf.clip_by_value(props[:,0],0,W-1)
            y1 = tf.clip_by_value(props[:,1],0,H-1)
            x2 = tf.clip_by_value(props[:,2],0,W-1)
            y2 = tf.clip_by_value(props[:,3],0,H-1)
            props = tf.stack([x1,y1,x2,y2],axis=1)

            k    = tf.minimum(self.pre_nms_top_n, tf.shape(sc)[0])
            idxs = tf.nn.top_k(sc,k=k).indices
            props = tf.gather(props, idxs)
            sc_sel= tf.gather(sc, idxs)

            keep = tf.image.non_max_suppression(
                props, sc_sel,
                max_output_size=self.post_nms_top_n,
                iou_threshold=self.nms_thresh
            )
            props = tf.gather(props,keep)
            pad   = self.post_nms_top_n - tf.shape(props)[0]
            props = tf.pad(props,[[0,pad],[0,0]])
            all_props.append(props)

        return tf.stack(all_props,axis=0)  # [B,post_nms_top_n,4]
