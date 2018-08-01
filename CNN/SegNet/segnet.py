import keras.backend as K
from functools import partial
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dropout, Input
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.layers.core import Activation, Flatten, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

class SegNet:
    """
    without pool indices
    """

    def __init__(self):
        pass


    def build_graph(self, input_shape=(360, 480, 3), classes=11):
        inputs = Input(shape=input_shape)

        # Encoder
        ## layer1
        enc_1 = self.encoder_layer(inputs, 64)

        ## layer2
        enc_2 = self.encoder_layer(enc_1, 128)

        ## layer3
        enc_3 = self.encoder_layer(enc_2, 256)

        ## layer4
        enc_4 = self.encoder_layer(enc_3, 512)

        ## lyaer5
        enc_5 = self.encoder_layer(enc_4, 512)

        # Decoder
        ## layer6
        dec_1 = self.decoder_layer(enc_5, 512)

        ## layer5
        dec_2 = self.decoder_layer(dec_1, 512)

        ## layer6
        dec_3 = self.decoder_layer(dec_2, 256)

        ## layer7
        dec_4 = self.decoder_layer(dec_3, 128)

        ## layer8
        dec_5 = self.decoder_layer(dec_4, 64)

        ## outputs
        outputs = self.output_layer(dec_5, input_shape, classes)

        ## model
        model = Model(inputs=inputs, outputs=outputs, name="SegNet")
        model.compile(loss="categorical_crossentropy",
                      optimizer="adadelta", metrics=["accuracy"])

        return model


    def encoder_layer(self, x, filters):
        x = self.helper_layer(x, filters)
        x = self.helper_layer(x, filters)
        if filters in (256, 512):
            x = self.helper_layer(x, filters)
        if filters != 512:
            x = MaxPooling2D(pool_size=(2, 2))(x)

        return x


    def decoder_layer(self, x, filters):
        if filters != 512:
            x = UpSampling2D(size=(2, 2))(x)
        x = self.helper_layer(x, filters)
        x = self.helper_layer(x, filters)
        if filters in (256, 512):
            x = self.helper_layer(x, filters)

        return x


    def helper_layer(self, x, filters):
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x


    def output_layer(self, x, input_size, classes):
        x = Conv2D(classes, (1, 1), padding="valid")(x)
        x = BatchNormalization()(x)
        x = Reshape((classes, input_size[0] * input_size[1]))(x)
        x = Permute((2, 1))(x)
        x = Activation("softmax")(x)

        return x