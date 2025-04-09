import tensorflow as tf
import numpy as np
import cv2
import os

class DavisDataset(tf.data.Dataset):
    """
    tf.data.Dataset for DAVIS:
      root_dir/
        train/ video1/images + video1/mask … video63
        val/   video64 … video76
        test/  video77 … video90

    Yields batches of:
      images: [B, H, W, 3], float32 [0,1]
      masks:  [B, H, W, 1], float32 {0,1}
      boxes:  [B, max_objects, 4], float32 [x1,y1,x2,y2]
      labels: [B, max_objects],      int32
    """
    def __new__(cls,
                root_dir,
                split,
                batch_size=2,
                shuffle=True,
                augment=True,
                target_size=(384, 640),
                max_objects=10,
                drop_remainder=True):
        base = os.path.join(root_dir, split)
        frame_paths, mask_paths = [], []

        # Gather file paths
        for video in sorted(os.listdir(base)):
            img_dir = os.path.join(base, video, "images")
            msk_dir = os.path.join(base, video, "mask")
            if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
                continue
            imgs = sorted(f for f in os.listdir(img_dir)
                          if f.lower().endswith(('.jpg','.png')))
            msks = sorted(f for f in os.listdir(msk_dir)
                          if f.lower().endswith('.png'))
            assert len(imgs)==len(msks), f"{video}: {len(imgs)} vs {len(msks)}"
            for im, mk in zip(imgs, msks):
                frame_paths.append(os.path.join(img_dir, im))
                mask_paths.append( os.path.join(msk_dir, mk))

        # Convert to tensors for numpy_function
        target_size_t = tf.constant(target_size, dtype=tf.int32)
        max_objs_t    = tf.constant(max_objects, dtype=tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((frame_paths, mask_paths))
        if shuffle:
            ds = ds.shuffle(min(len(frame_paths), 1000),
                            reshuffle_each_iteration=True)

        ds = ds.map(
            lambda fp, mp: tf.numpy_function(
                func=cls._load_and_preprocess,
                inp=[fp, mp, target_size_t, max_objs_t],
                Tout=[tf.float32, tf.float32, tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if augment:
            ds = ds.map(cls._augment, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.padded_batch(
            batch_size,
            padded_shapes=(
                [*target_size, 3],
                [*target_size, 1],
                [max_objects, 4],
                [max_objects]
            ),
            padding_values=(0.0, 0.0, 0.0, 0),
            drop_remainder=drop_remainder
        )

        return ds.prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def _load_and_preprocess(frame_path, mask_path, target_size, max_objects):
        frame_path = frame_path.decode()
        mask_path  = mask_path.decode()
        height = int(target_size[0]); width  = int(target_size[1])
        max_objects = int(max_objects)

        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid target_size: {target_size}")

        # Load & resize image
        img = cv2.imread(frame_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {frame_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        # Load & resize mask
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        m = cv2.resize(m, (width, height),
                       interpolation=cv2.INTER_NEAREST)
        mask = (m > 127).astype(np.float32)[..., None]

        # Connected‐component boxes
        ys, xs = np.where(mask[...,0]>0)
        boxes, labels = [], []
        if ys.size:
            binary = np.uint8(m > 127)
            num_labels, lbl_im = cv2.connectedComponents(binary)
            for lab in range(1, min(num_labels, max_objects+1)):
                y_i, x_i = np.where(lbl_im==lab)
                if y_i.size>10:
                    y1,y2 = y_i.min(), y_i.max()
                    x1,x2 = x_i.min(), x_i.max()
                    boxes.append([x1,y1,x2,y2])
                    labels.append(1)

        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            labels= np.array(labels, dtype=np.int32)
        else:
            boxes = np.zeros((0,4), dtype=np.float32)
            labels= np.zeros((0,),   dtype=np.int32)

        # Pad/truncate
        if boxes.shape[0]>max_objects:
            boxes = boxes[:max_objects]; labels=labels[:max_objects]
        elif boxes.shape[0]<max_objects:
            pad_n = max_objects - boxes.shape[0]
            boxes  = np.vstack([boxes, np.zeros((pad_n,4),dtype=np.float32)])
            labels = np.concatenate([labels, np.zeros((pad_n,),dtype=np.int32)])

        return img, mask, boxes, labels

    @staticmethod
    def _augment(img, mask, boxes, labels):
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, mask, boxes, labels
    