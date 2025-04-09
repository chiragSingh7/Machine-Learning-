# eval.py

import os
# ── FORCE CPU & SILENCE INFO LOGS ──
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import tensorflow as tf
import numpy as np
from data.dataset import DavisDataset
from model.backbone import get_backbone
from model.faster_rcnn import FasterRCNN
from utils.metrics import f_measure, s_measure, mean_absolute_error

def main(args):
    # Build the dataset
    ds = DavisDataset(
        root_dir=args.data_root,
        split=args.split,
        batch_size=1,
        shuffle=False,
        augment=False,
        target_size=(args.height, args.width),
        drop_remainder=False
    )

    # Build the backbone & model
    backbone = get_backbone(input_shape=(args.height, args.width, 3))
    model = FasterRCNN(backbone,
                       num_classes=args.num_classes,
                       image_shape=(args.height, args.width))

    # Restore checkpoint
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(args.checkpoint_dir)) \
        .expect_partial()

    # Metrics accumulators
    fs, ss, ms = [], [], []

    for imgs, masks, gt_boxes, _ in ds:
        dets, _ = model(imgs,
                        gt_boxes=gt_boxes,
                        gt_masks=masks,
                        training=False)

        # Extract outputs
        props      = dets['proposals'][0].numpy()      # [P,4]
        cls_logits = dets['class_logits'].numpy()      # [P,2]
        mask_preds = dets['mask_preds'].numpy()        # [P,Hf,Wf,2]

        # Compute scores & pick best foreground
        probs   = tf.nn.softmax(cls_logits, axis=-1).numpy()
        scores  = probs[:,1]
        classes = np.argmax(cls_logits, axis=-1)
        fg_idxs = np.where(classes == 1)[0]

        if fg_idxs.size > 0:
            best = fg_idxs[np.argmax(scores[fg_idxs])]
            m_small = mask_preds[best, ..., 1]  # [Hf,Wf]
            # Resize back to full size
            m_resized = tf.image.resize(
                m_small[..., None],
                (args.height, args.width),
                method='bilinear'
            )[...,0].numpy()
            pred_mask = (m_resized > 0.5).astype(np.uint8)
        else:
            pred_mask = np.zeros((args.height, args.width), dtype=np.uint8)

        # Ground‑truth mask
        gt_mask = masks[0, ..., 0].numpy().astype(np.uint8)

        # Compute metrics
        fs.append(f_measure(pred_mask, gt_mask))
        ss.append(s_measure(pred_mask, gt_mask))
        ms.append(mean_absolute_error(pred_mask, gt_mask))

    # Print final scores
    print(f"F‑measure: {np.mean(fs):.4f}")
    print(f"S‑measure: {np.mean(ss):.4f}")
    print(f"MAE:        {np.mean(ms):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",      required=True,
                        help="Path to DAVIS root (contains train/ val/ test/)")
    parser.add_argument("--split",          required=True,
                        choices=["train","val","test"],
                        help="Which split to evaluate")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing your saved checkpoints")
    parser.add_argument("--height", type=int, default=384,
                        help="Frame height for resizing (default 384)")
    parser.add_argument("--width",  type=int, default=640,
                        help="Frame width for resizing (default 640)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of object classes (including background)")
    args = parser.parse_args()
    main(args)
