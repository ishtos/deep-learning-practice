from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

## This function from https://github.com/keras-team/keras/blob/master/keras/utils/conv_utils.py
def normalize_tuple(value, n, name):
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except ValueError:
                raise ValueError('The `' + name + '` argument must be a tuple of ' +
                                 str(n) + ' integers. Received: ' +
                                 str(value) + ' '
                                 'including element ' + str(single_value) + ' of type' +
                                 ' ' + str(type(single_value)))
    return value_tuple


## This function from https://github.com/keras-team/keras/blob/master/keras/utils/conv_utils.py
def normalize_padding(value):
    padding = value.lower()
    allowed = {'valid', 'same', 'causal'}
    if K.backend() == 'theano':
        allowed.add('full')
    if padding not in allowed:
        raise ValueError('The `padding` argument must be one of "valid", "same" (or "causal" for Conv1D). '
                         'Received: ' + str(padding))
    return padding


## This function from https://github.com/keras-team/keras/blob/master/keras/utils/conv_utils.py
def conv_output_length(input_length, filter_size, stride, padding, dilation=1):
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


# This class inpired from https://github.com/ykamikawa/SegNet/blob/master/Mylayers.py
class AdvancedMaxPooling2D(Layer):

    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        self.pool_size = normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = normalize_tuple(strides, 2, 'strides')
        self.padding = normalize_padding(padding)
        super(AdvancedMaxPooling2D, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides

        ksize = [1, pool_size[0], pool_size[1], 1]
        strides = [1, strides[0], strides[1], 1]
        padding = padding.upper()
        
        output, argmax = K.tf.nn.max_pool_with_argmax(input=inputs, ksize=ksize, strides=strides, padding=padding)
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        rows = input_shape[1]
        cols = input_shape[2]
        rows = conv_output_length(rows, self.pool_size[0], self.strides[0], self.padding)
        cols = conv_output_length(cols, self.pool_size[1], self.strides[1], self.padding)
        
        # same shaep
        output_shape = (input_shape[0], rows, cols, input_shape[3])
        argmax_shape = (input_shape[0], rows, cols, input_shape[3])

        return [output_shape, argmax_shape]


# This class inpired from https://github.com/ykamikawa/SegNet/blob/master/Mylayers.py
class AdvancedUpSampling2D(Layer):

    def __init__(self, size=(2, 2), **kwargs):
        self.size = size
        super(AdvancedUpSampling2D, self).__init__(**kwargs)

    def call(self, inputs):
        updates, mask = inputs[0], inputs[1]
       
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            output_shape = (input_shape[0], input_shape[1] * self.size[0],
                            input_shape[2] * self.size[1], input_shape[3])

            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(K.tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
            
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            c = one_like_mask * K.arange(output_shape[3], dtype='int32')

            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, c]), [4, updates_size]))
            updates = K.reshape(updates, [updates_size])
            res = K.tf.scatter_nd(indices=indices, updates=updates, shape=output_shape)
            return res

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        height = self.size[0] * mask_shape[1] if mask_shape[1] is not None else None
        width = self.size[1] * mask_shape[2] if mask_shape[2] is not None else None
        
        return (mask_shape[0], height, width, mask_shape[3])
