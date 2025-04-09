import argparse
import os
import tensorflow as tf
import numpy as np
from data.dataset import DavisDataset
from model.backbone import get_backbone
from model.faster_rcnn import FasterRCNN
from utils.metrics import f_measure, s_measure, mean_absolute_error

# Optionally disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
    except RuntimeError as e:
        print(e)

def main(args):
    # Dataset
    train_ds = DavisDataset(
        root_dir=args.data_root, split='train',
        batch_size=args.batch_size, shuffle=True, augment=True,
        target_size=(384, 640)
    )
    val_ds = DavisDataset(
        root_dir=args.data_root, split='val',
        batch_size=1, shuffle=False, augment=False,
        target_size=(384, 640)
    )

    # Backbone & Model
    backbone = get_backbone(input_shape=(384, 640, 3))
    model = FasterRCNN(backbone, num_classes=2, image_shape=(384, 640))
    opt = tf.keras.optimizers.Adam(args.lr)

    # Checkpointing
    ckpt = tf.train.Checkpoint(model=model, optimizer=opt)
    mgr = tf.train.CheckpointManager(ckpt, args.checkpoint_dir, max_to_keep=3)
    writer = tf.summary.create_file_writer(args.log_dir)

    if args.resume and mgr.latest_checkpoint:
        print(f"Restoring from checkpoint: {mgr.latest_checkpoint}")
        ckpt.restore(mgr.latest_checkpoint)

    steps_per_epoch = sum(1 for _ in train_ds)
    print(f"Steps per epoch: {steps_per_epoch}")

    best_val_f = 0.0
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
        total_loss = 0.0
        step = 0
        error_count = 0

        for batch_idx, (imgs, masks, gt_boxes, gt_labels) in enumerate(train_ds):
            try:
                with tf.GradientTape() as tape:
                    dets, losses = model(imgs, gt_boxes=gt_boxes, gt_masks=masks, training=True)
                    loss = tf.add_n(list(losses.values())) / args.accumulation_steps

                grads = tape.gradient(loss, model.trainable_variables)
                accumulated_gradients = [
                    (accum + grad if grad is not None else accum)
                    for accum, grad in zip(accumulated_gradients, grads)
                ]
                total_loss += loss * args.accumulation_steps
                step += 1

                if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == steps_per_epoch:
                    opt.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
                    accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]

                    if step % args.log_interval == 0:
                        avg_loss = total_loss / step
                        print(f"Step {batch_idx+1}/{steps_per_epoch}, Loss: {avg_loss:.4f}")
                        with writer.as_default():
                            tf.summary.scalar('train_loss', avg_loss, step=epoch * steps_per_epoch + batch_idx)
                            for name, val in losses.items():
                                tf.summary.scalar(f'train_{name}', val, step=epoch * steps_per_epoch + batch_idx)

            except tf.errors.ResourceExhaustedError as e:
                print(f"OOM error at batch {batch_idx}: {e}")
                error_count += 1
                if error_count >= args.max_batch_errors:
                    print("Too many errors, skipping epoch.")
                    return
                continue
            except Exception as e:
                print(f"Error at batch {batch_idx}: {e}")
                error_count += 1
                if error_count >= args.max_batch_errors:
                    print("Too many errors, skipping epoch.")
                    return
                continue

        # Validation
        fs, ss, ms = [], [], []
        for imgs, masks, gt_boxes, _ in val_ds:
            try:
                dets, _ = model(imgs, gt_boxes=gt_boxes, gt_masks=masks, training=False)
                cls_logits = dets['class_logits'].numpy()
                probs = tf.nn.softmax(cls_logits, axis=-1).numpy()
                scores = probs[:, 1]
                classes = np.argmax(cls_logits, axis=-1)
                mask_preds = dets['mask_preds'].numpy()
                fg = np.where(classes == 1)[0]

                if fg.size > 0:
                    best = fg[np.argmax(scores[fg])]
                    m_small = mask_preds[best, :, :, 1]
                    m_resized = tf.image.resize(m_small[..., None], (384, 640), method='bilinear')[..., 0].numpy()
                    pred_mask = (m_resized > 0.5).astype(np.uint8)
                else:
                    pred_mask = np.zeros((384, 640), dtype=np.uint8)

                gt_mask = masks[0, ..., 0].numpy().astype(np.uint8)
                fs.append(f_measure(pred_mask, gt_mask))
                ss.append(s_measure(pred_mask, gt_mask))
                ms.append(mean_absolute_error(pred_mask, gt_mask))

            except Exception as e:
                print(f"Validation error: {e}")
                continue

        val_f = np.mean(fs) if fs else 0.0
        val_s = np.mean(ss) if ss else 0.0
        val_mae = np.mean(ms) if ms else 0.0
        print(f"Validation -> F: {val_f:.4f}, S: {val_s:.4f}, MAE: {val_mae:.4f}")

        with writer.as_default():
            tf.summary.scalar('val_f_measure', val_f, step=(epoch + 1) * steps_per_epoch)
            tf.summary.scalar('val_s_measure', val_s, step=(epoch + 1) * steps_per_epoch)
            tf.summary.scalar('val_mae', val_mae, step=(epoch + 1) * steps_per_epoch)

        # Checkpointing
        save_path = mgr.save()
        print(f"Checkpoint saved: {save_path}")

        if val_f > best_val_f:
            best_val_f = val_f
            no_improve_epochs = 0
            print(f"New best F-measure: {val_f:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No F-measure improvement for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= args.patience:
            print("Early stopping triggered.")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--max_batch_errors', type=int, default=5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args)
