import numpy as np
import tensorflow as tf

def f_measure(pred, gt, beta=1.0, eps=1e-7):
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = np.logical_and(pred,gt).sum()
    fp = np.logical_and(pred,~gt).sum()
    fn = np.logical_and(~pred,gt).sum()
    precision = tp/(tp+fp+eps)
    recall    = tp/(tp+fn+eps)
    return (1+beta**2)*(precision*recall)/(beta**2*precision+recall+eps)

def s_measure(pred, gt):
    pred_tf = tf.convert_to_tensor(pred[...,None],dtype=tf.float32)
    gt_tf   = tf.convert_to_tensor(gt[...,None],  dtype=tf.float32)
    return float(tf.image.ssim(pred_tf, gt_tf, max_val=1.0).numpy())

def mean_absolute_error(pred, gt):
    return float(np.mean(np.abs(pred-gt)))

def accuracy_score(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    return np.mean(pred==gt)

def iou_score(pred, gt, eps=1e-7):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = np.logical_and(pred,gt).sum()
    union = np.logical_or(pred,gt).sum()
    return inter/(union+eps)
