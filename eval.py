import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score

import tensorflow as tf
from data.dataset import DavisDataset
from model.backbone import get_backbone
from model.faster_rcnn import FasterRCNN
from utils.metrics import (
    f_measure, s_measure, mean_absolute_error,
    accuracy_score, iou_score
)

def main(args):
    ds = DavisDataset(
        root_dir=args.data_root, split=args.split,
        batch_size=1, shuffle=False, augment=False,
        target_size=(args.height,args.width),
        drop_remainder=False
    )

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

    fs, ss, ms = [], [], []
    accs, ious, precs, recs, f1s = [], [], [], [], []
    all_probs, all_gt = [], []

    for imgs, masks, _, _ in ds:
        dets, _ = model(imgs, training=False)
        boxes  = dets['boxes'].numpy()
        scores = dets['scores'].numpy()
        masks_pred = dets['masks'].numpy()

        if boxes.shape[0]>0:
            idx = np.argmax(scores)
            prob_small = masks_pred[idx]
            prob_full  = tf.image.resize(
                prob_small[...,None],
                (args.height,args.width),
                method='bilinear'
            )[...,0].numpy()
            pred_mask = (prob_full>args.mask_thresh).astype(np.uint8)
            all_probs.append(prob_full.flatten())
        else:
            pred_mask = np.zeros((args.height,args.width),dtype=np.uint8)
            all_probs.append(np.zeros(args.height*args.width))

        gt_mask = masks[0,...,0].numpy().astype(np.uint8)
        all_gt.append(gt_mask.flatten())

        fs.append(f_measure(pred_mask,gt_mask))
        ss.append(s_measure(pred_mask,gt_mask))
        ms.append(mean_absolute_error(pred_mask,gt_mask))
        accs.append(accuracy_score(pred_mask,gt_mask))
        ious.append(iou_score(pred_mask,gt_mask))

        flat_p = pred_mask.flatten(); flat_g = gt_mask.flatten()
        precs.append(precision_score(flat_g,flat_p,zero_division=0))
        recs.append(recall_score(flat_g,flat_p,zero_division=0))
        f1s.append(f1_score(flat_g,flat_p,zero_division=0))

    all_probs = np.concatenate(all_probs)
    all_gt    = np.concatenate(all_gt)
    fpr,tpr,_ = roc_curve(all_gt,all_probs)
    roc_auc   = auc(fpr,tpr)

    plt.figure()
    plt.plot(fpr,tpr,label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    out_png = os.path.join(args.checkpoint_dir,"roc_curve.png")
    plt.savefig(out_png); plt.close()
    print(f"Saved ROC → {out_png}")

    print(f"F‑measure: {np.mean(fs):.4f}")
    print(f"S‑measure: {np.mean(ss):.4f}")
    print(f"MAE:       {np.mean(ms):.4f}")
    print(f"Accuracy:  {np.mean(accs):.4f}")
    print(f"IoU:       {np.mean(ious):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall:    {np.mean(recs):.4f}")
    print(f"F1‑score:  {np.mean(f1s):.4f}")
    print(f"AUC:       {roc_auc:.4f}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      required=True)
    p.add_argument("--split",          choices=["train","val","test"],required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--height",   type=int,   default=384)
    p.add_argument("--width",    type=int,   default=640)
    p.add_argument("--num_classes",type=int,  default=2)
    p.add_argument("--score_thresh", type=float, default=0.5)
    p.add_argument("--iou_thresh",   type=float, default=0.5)
    p.add_argument("--mask_thresh",  type=float, default=0.5)
    args = p.parse_args()
    main(args)
