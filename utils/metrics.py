import numpy as np
from skimage.metrics import structural_similarity as ssim

def f_measure(pred_mask, gt_mask, beta=1):
    pred = (pred_mask>0.5).astype(np.uint8)
    gt   = (gt_mask>0.5).astype(np.uint8)
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    P  = tp/(tp+fp+1e-8)
    R  = tp/(tp+fn+1e-8)
    return (1+beta**2)*P*R/(beta**2*P+R+1e-8)

def s_measure(pred_mask, gt_mask):
    pm = (pred_mask*255).astype(np.uint8)
    gm = (gt_mask*255).astype(np.uint8)
    return ssim(gm, pm, data_range=255)

def mean_absolute_error(pred_mask, gt_mask):
    return np.mean(np.abs(pred_mask-gt_mask))
