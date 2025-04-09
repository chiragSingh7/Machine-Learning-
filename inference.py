# inference.py

import os
# ── FORCE CPU & SILENCE INFO LOGS ──
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import cv2
import numpy as np
import tensorflow as tf
from model.backbone import get_backbone
from model.faster_rcnn import FasterRCNN
from utils.visualizations import draw_boxes, draw_mask

def main(args):
    # Build backbone & model
    backbone = get_backbone(input_shape=(args.height, args.width, 3))
    model = FasterRCNN(
        backbone,
        num_classes=args.num_classes,
        image_shape=(args.height, args.width)
    )
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(args.checkpoint_dir)) \
        .expect_partial()

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W0  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W0, H0)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (args.width, args.height))
        small = small.astype(np.float32) / 255.0
        inp   = tf.expand_dims(small, 0)  # [1,H,W,3]

        # Inference
        dets, _ = model(inp, training=False)
        props      = dets['proposals'][0].numpy()    # [P,4]
        cls_logits = dets['class_logits'].numpy()    # [P, num_classes]
        mask_preds = dets['mask_preds'].numpy()      # [P, Hf, Wf, num_classes]

        # Score & pick best foreground
        probs   = tf.nn.softmax(cls_logits, axis=-1).numpy()
        scores  = probs[:,1]
        classes = np.argmax(cls_logits, axis=-1)
        fg_idxs = np.where(classes == 1)[0]

        vis = frame.copy()
        if fg_idxs.size > 0:
            best = fg_idxs[np.argmax(scores[fg_idxs])]
            box  = props[best]                     # [x1,y1,x2,y2]

            # Resize mask back to model input size
            m_small = mask_preds[best, ..., 1]     # [Hf,Wf]
            m_small = (m_small > 0.5).astype(np.uint8)
            m_small = cv2.resize(
                m_small,
                (args.width, args.height),
                interpolation=cv2.INTER_NEAREST
            )
            # Resize mask to original video resolution
            mask_large = cv2.resize(
                m_small,
                (W0, H0),
                interpolation=cv2.INTER_NEAREST
            )

            # Scale box to original resolution
            x1, y1, x2, y2 = box
            sx = W0 / args.width
            sy = H0 / args.height
            x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

            # Draw
            vis = draw_boxes(
                vis,
                np.array([[x1, y1, x2, y2]]),
                labels=[1],
                scores=[scores[best]]
            )
            vis = draw_mask(vis, mask_large)

        writer.write(vis)

    cap.release()
    writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",     required=True,
                        help="Path to input video file")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing model checkpoints")
    parser.add_argument("--output_path",    required=True,
                        help="Path to write output video")
    parser.add_argument("--height",  type=int, default=384,
                        help="Inference frame height (must match training)")
    parser.add_argument("--width",   type=int, default=640,
                        help="Inference frame width (must match training)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes (including background)")
    args = parser.parse_args()
    main(args)
