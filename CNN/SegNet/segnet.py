import keras.backend as K
from functools import partial
from keras.models import Model, Input
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dropout
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.layers.core import Activation, Flatten, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

class SegNet:
    """
    without pool indices
    """

    def __init__(self, input_size=(360, 480, 3), classes=11):
        self.input_size = input_size
        inputs = Input(input_size)

        # Encoder
        ## layer1
        enc1 = self.encoder_layer(inputs, 64)

        ## layer2
        enc2 = self.encoder_layer(enc1, 128)

        ## layer3
        enc3 = self.encoder_layer(enc2, 256)

        ## layer4
        enc4 = self.encoder_layer(enc3, 512)

        ## lyaer5
        enc5 = self.encoder_layer(enc4, 512)

        # Decoder
        ## layer6 
        dec0 = self.decoder_layer(enc5, 512)

        ## layer5
        dec1 = self.decoder_layer(dec0, 512)

        ## layer6
        dec2 = self.decoder_layer(dec1, 256)

        ## layer7
        dec3 = self.decoder_layer(dec2, 128)

        ## layer8
        dec4 = self.decoder_layer(dec3, 64)

        ## outputs
        outputs = self.output_layer(dec4, input_size, classes)
        
        ## model
        model = Model(inputs, outputs)
        model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

        self.model = model

    def encoder_layer(self, x, filters):
        if filters == 64: 
            x = Conv2D(filters, (3, 3), padding="same", input_shape=(360, 480))(x)
        else:
            x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        if filters in (256, 512):
            x = Conv2D(filters, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        if filters != 512:
            x = MaxPooling2D(pool_size=(2, 2))(x)

        return x


    def decoder_layer(self, x, filters):
        if filters != 512:
            x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        if filters in (256, 512):
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

    
    def segnet(self):
        return self.model
