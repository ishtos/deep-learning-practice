from functools import partial
import tensorflow as tf


def build(inputs, labels, fcn_classes):
    my_conv2d_layer = partial(tf.nn.conv2d,
                              filters=64,
                              kernel_size=3,
                              strides=1,
                              activation=tf.nn.relu,
                              padding='same')

    my_deconv2d_layer = partial(tf.nn.conv2d,
                                filters=fcn_classes,
                                kernel_size=4,
                                strides=2,
                                padding='valid')

    my_max_pooling2d_layer = partial(tf.nn.max_pooling2d,
                                    filters=2,
                                    strides=2)

    conv1_1 = my_conv2d_layer(inputs)
    conv1_2 = my_conv2d_layer(conv1_1)
    max_pool1 = my_max_pooling2d_layer(conv1_2)

    conv2_1 = my_conv2d_layer(max_pool1, filters=128)
    conv2_2 = my_conv2d_layer(conv2_1, filters=128)
    max_pool2 = my_max_pooling2d_layer(conv2_2)

    conv3_1 = my_conv2d_layer(max_pool2, filters=256)
    conv3_2 = my_conv2d_layer(conv3_1, filters=256)
    conv3_3 = my_conv2d_layer(conv3_1, filters=256)
    max_pool3 = my_max_pooling2d_layer(conv3_3)

    conv4_1 = my_conv2d_layer(max_pool3, filters=512)
    conv4_2 = my_conv2d_layer(conv4_1, filters=512)
    conv4_3 = my_conv2d_layer(conv4_3, filters=512)
    max_pool4 = my_max_pooling2d_layer(conv4_2)

    conv5_1 = my_conv2d_layer(max_pool4, filters=512)
    conv5_2 = my_conv2d_layer(conv5_1, filters=512)
    conv5_3 = my_conv2d_layer(conv5_2, filters=512)
    max_pool5 = my_max_pooling2d_layer(conv5_3)

    pool3 = my_conv2d_layer(max_pool3, filters=fcn_classes, kernel_size=1, padding='valid')
    pool4 = my_conv2d_layer(max_pool4, filters=fcn_classes, kernel_size=1, padding='valid')
    pool5 = my_conv2d_layer(max_pool5, filters=fcn_classes, kernel_size=1, padding='valid')

    deconv4 = my_deconv2d_layer(pool4, filters=fcn_classes, kernel_size=4, strides=2)
    deconv4 = tf.keras.layers.Cropping2D(deconv4, (2, 2), (2, 2))

    deconv5 = my_deconv2d_layer(pool5, filters=fcn_classes, skernel_size=8, strides=4)
    deconv5 = tf.keras.layers.Cropping2D(deconv4, (2, 2), (2, 2))

    merge = tf.add(pool3, deconv4, deconv5)
    deconv = my_deconv2d_layer(merge, filters=fcn_classes, skernel_size=16, strides=8)
    deconv = tf.keras.layers.Cropping2D(deconv, (4, 4), (4, 4))

    output = tf.reshape(deconv, fcn_classes, 224*224)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output, dim=fcn_classes)