import tensorflow as tf

def get_backbone(input_shape=(384,640,3)):
    """
    ResNet50 up to Conv5_x (C5), pretrained on ImageNet.
    """
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    c5 = base.get_layer('conv5_block3_out').output
    return tf.keras.Model(inputs=base.input, outputs=c5, name='resnet50_c5')
