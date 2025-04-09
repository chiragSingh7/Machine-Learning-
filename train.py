import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import argparse
import datetime
import tensorflow as tf
from data.dataset import DavisDataset
from model.backbone import get_backbone
from model.faster_rcnn import FasterRCNN

def lr_schedule(epoch, lr):
    return lr if epoch<10 else lr*tf.math.exp(-0.1)

def main(args):
    train_ds = DavisDataset(
        root_dir=args.data_root, split='train',
        batch_size=args.batch_size, shuffle=True, augment=True,
        target_size=(args.height,args.width), drop_remainder=True
    )
    val_ds = DavisDataset(
        root_dir=args.data_root, split='val',
        batch_size=1, shuffle=False, augment=False,
        target_size=(args.height,args.width), drop_remainder=False
    )

    backbone = get_backbone(input_shape=(args.height,args.width,3))
    model    = FasterRCNN(
        backbone,
        num_classes=args.num_classes,
        image_shape=(args.height,args.width),
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh
    )

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.lr, momentum=0.9
    )
    lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    ckpt = tf.train.Checkpoint(model=model,optimizer=optimizer)
    mgr  = tf.train.CheckpointManager(
        ckpt,args.checkpoint_dir,max_to_keep=3
    )
    logdir = os.path.join(
        args.log_dir,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tb_cb = tf.keras.callbacks.TensorBoard(logdir)

    for epoch in range(args.epochs):
        train_loss = 0.0
        for step,(imgs,masks,gt_boxes,_) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                _,losses = model(
                    imgs,gt_boxes=gt_boxes,gt_masks=masks,training=True
                )
                total_loss = (
                    losses['rpn_class_loss']
                    + args.lambda_rpn * losses['rpn_bbox_loss']
                    + losses['rcnn_class_loss']
                    + args.lambda_rcnn * losses['rcnn_bbox_loss']
                    + losses['mask_loss']
                )
            grads = tape.gradient(total_loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            train_loss += total_loss.numpy()
        train_loss /= (step+1)

        val_loss = 0.0
        for step,(imgs,masks,gt_boxes,_) in enumerate(val_ds):
            _,losses = model(
                imgs,gt_boxes=gt_boxes,gt_masks=masks,training=False
            )
            total_loss = (
                losses['rpn_class_loss']
                + args.lambda_rpn * losses['rpn_bbox_loss']
                + losses['rcnn_class_loss']
                + args.lambda_rcnn * losses['rcnn_bbox_loss']
                + losses['mask_loss']
            )
            val_loss += total_loss.numpy()
        val_loss /= (step+1)

        with tb_cb.writer.as_default():
            tf.summary.scalar("train_loss",train_loss,step=epoch)
            tf.summary.scalar("val_loss",  val_loss,  step=epoch)
            tf.summary.scalar("lr",        optimizer.lr(epoch),step=epoch)

        print(f"Epoch {epoch+1}/{args.epochs} — "
              f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        mgr.save()
        lr_cb.on_epoch_end(epoch)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--log_dir",        required=True)
    p.add_argument("--batch_size",     type=int, default=2)
    p.add_argument("--epochs",         type=int, default=20)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--height",         type=int, default=384)
    p.add_argument("--width",          type=int, default=640)
    p.add_argument("--num_classes",    type=int, default=2)
    p.add_argument("--score_thresh",   type=float, default=0.5)
    p.add_argument("--iou_thresh",     type=float, default=0.5)
    p.add_argument("--lambda_rpn",     type=float, default=1.0)
    p.add_argument("--lambda_rcnn",    type=float, default=1.0)
    args = p.parse_args()
    main(args)
