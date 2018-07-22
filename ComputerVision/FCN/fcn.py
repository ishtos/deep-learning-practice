from functools import partial
import tensorflow as tf


def build(self):
    my_conv2d_layer = partial(tf.layers.conv2d,
                            filters=3,

                            activation=tf.nn.relu,
                            padding='same')

    my_max_pooling2d_layer = partial(tf.layers.max_pooling2d,
                                    filters=2,
                                    strides=[2,2])

    conv1_1 = my_conv2d_layer(inputs)
    conv1_2 = my_conv2d_layer(conv1_1)
    max_pool1 = my_max_pooling2d_layer(conv1_2)

    conv2_1 = my_conv2d_layer(max_pool1)
    conv2_2 = my_conv2d_layer(conv2_1)
    max_pool2 = my_max_pooling2d_layer(conv2_2)

    conv3_1 = my_conv2d_layer(max_pool2)
    conv3_2 = my_conv2d_layer(conv3_1)
    conv3_3 = my_conv2d_layer(conv3_2)
    max_pool3 = my_max_pooling2d_layer(conv3_3)

    conv4_1 = my_conv2d_layer(max_pool3)
    conv4_2 = my_conv2d_layer(conv4_1)
    conv4_3 = my_conv2d_layer(conv4_2)
    max_pool4 = my_max_pooling2d_layer(conv4_3)

    conv5_1 = my_conv2d_layer(max_pool4)
    conv5_2 = my_conv2d_layer(conv5_1)
    conv5_3 = my_conv2d_layer(conv5_2)
    max_pool5 = my_max_pooling2d_layer(conv5_3)

    conv6 = my_conv2d_layer(max_pool5)
    conv6 = tf.layers.dropout(conv6, rate=0.5)
    
    conv7 = my_conv2d_layer(con6, filters=1)
