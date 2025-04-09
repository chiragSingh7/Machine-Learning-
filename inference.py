import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import argparse
import cv2
import numpy as np
import tensorflow as tf
from model.backbone import get_backbone
from model.faster_rcnn import FasterRCNN
from utils.visualizations import draw_boxes, draw_mask

def main(args):
    backbone = get_backbone(input_shape=(args.height,args.width,3))
    model    = FasterRCNN(
        backbone,
        num_classes=args.num_classes,
        image_shape=(args.height,args.width),
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh
    )
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(args.checkpoint_dir))\
        .expect_partial()

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W0  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        args.output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W0,H0)
    )

    while True:
        ret,frame = cap.read()
        if not ret: break

        rgb   = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb,(args.width,args.height))
        small = small.astype(np.float32)/255.0
        inp   = tf.expand_dims(small,0)

        dets,_ = model(inp,training=False)
        boxes  = dets['boxes'].numpy()
        scores = dets['scores'].numpy()
        masks  = dets['masks'].numpy()

        vis = frame.copy()
        if boxes.shape[0]>0:
            idx = np.argmax(scores)
            x1,y1,x2,y2 = boxes[idx]
            sx,sy = W0/args.width, H0/args.height
            x1,y1,x2,y2 = (int(x1*sx),int(y1*sy),
                          int(x2*sx),int(y2*sy))

            m_small = masks[idx]
            m_small = (m_small>0.5).astype(np.uint8)
            mask_large = cv2.resize(
                m_small,(args.width,args.height),
                interpolation=cv2.INTER_NEAREST
            )
            mask_large = cv2.resize(
                mask_large,(W0,H0),
                interpolation=cv2.INTER_NEAREST
            )

            vis = draw_boxes(vis,np.array([[x1,y1,x2,y2]]),
                             labels=[1],scores=[scores[idx]])
            vis = draw_mask(vis,mask_large)

        writer.write(vis)

    cap.release()
    writer.release()

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video_path",     required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_path",    required=True)
    p.add_argument("--height",    type=int, default=384)
    p.add_argument("--width",     type=int, default=640)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--score_thresh", type=float, default=0.5)
    p.add_argument("--iou_thresh",   type=float, default=0.5)
    args = p.parse_args()
    main(args)
