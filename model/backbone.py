# model/backbone.py

import tensorflow as tf

def get_backbone(input_shape=(240,426,3)):
    """
    A lighter ResNet50‑C5 backbone for CPU training on 240×426 inputs.
    """
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    c5 = base.get_layer('conv5_block3_out').output
    return tf.keras.Model(inputs=base.input, outputs=c5, name='resnet50_c5')
