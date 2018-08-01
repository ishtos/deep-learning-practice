from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D 
from keras.layers.normalization import BatchNormalization

from AdvancedLayers import AdvancedMaxPooling2D, AdvancedUpSampling2D


class SegNet:
    
    def __init__(self):
        pass


    def build_graph(self, input_shape=(360, 480, 3), n_classes=11, kernel=3, pool_size=(2, 2)):
        inputs = Input(shape=input_shape)

        ## Encoder
        conv_1 = self.my_layer(inputs, 64)
        conv_2 = self.my_layer(conv_1, 64)

        pool_1, mask_1 = AdvancedMaxPooling2D(pool_size)(conv_2)

        conv_3 = self.my_layer(pool_1, 128)
        conv_4 = self.my_layer(conv_3, 128)

        pool_2, mask_2 = AdvancedMaxPooling2D(pool_size)(conv_4)

        conv_5 = self.my_layer(pool_2, 256)
        conv_6 = self.my_layer(conv_5, 256)
        conv_7 = self.my_layer(conv_6, 256)

        pool_3, mask_3 = AdvancedMaxPooling2D(pool_size)(conv_7)

        conv_8 = self.my_layer(pool_3, 512)
        conv_9 = self.my_layer(conv_8, 512)
        conv_10 = self.my_layer(conv_9, 512)

        pool_4, mask_4 = AdvancedMaxPooling2D(pool_size)(conv_10)

        conv_11 = self.my_layer(pool_4, 512)
        conv_12 = self.my_layer(conv_11, 512)
        conv_13 = self.my_layer(conv_12, 512)

        pool_5, mask_5 = AdvancedMaxPooling2D(pool_size)(conv_13)

        ## Decoder
        upsamp_1 = AdvancedUpSampling2D(pool_size)([pool_5, mask_5])

        conv_14 = self.my_layer(upsamp_1, 512)
        conv_15 = self.my_layer(conv_14, 512)
        conv_16 = self.my_layer(conv_15, 512)

        upsamp_2 = AdvancedUpSampling2D(pool_size)([conv_16, mask_4])

        conv_17 = self.my_layer(upsamp_2, 256)
        conv_18 = self.my_layer(conv_17, 256)
        conv_19 = self.my_layer(conv_18, 256)

        upsamp_3 = AdvancedUpSampling2D(pool_size)([conv_19, mask_3])

        conv_20 = self.my_layer(upsamp_3, 256)
        conv_21 = self.my_layer(conv_20, 256)
        conv_22 = self.my_layer(conv_21, 256)

        upsamp_4 = AdvancedUpSampling2D(pool_size)([conv_22, mask_2])

        conv_23 = self.my_layer(upsamp_4, 128)
        conv_24 = self.my_layer(conv_23, 64)

        upsamp_5 = AdvancedUpSampling2D(pool_size)([conv_24, mask_1])

        conv_25 = self.my_layer(upsamp_5, 64)

        conv_26 = Convolution2D(n_classes, (1, 1), padding="valid")(conv_25)
        conv_26 = BatchNormalization()(conv_26)
        conv_26 = Reshape(target_shape=(input_shape[0] * input_shape[1], n_classes),
                          input_shape=(input_shape[0], input_shape[1], n_classes))(conv_26)

        outputs = Activation("softmax")(conv_26)

        model = Model(inputs=inputs, outputs=outputs, name="SegNet")
        model.compile(loss="categorical_crossentropy",
                      optimizer='adadelta', metrics=["accuracy"])

        return model


    def my_layer(self, x, filters, kernel=(3,3), padding='same'):  
        x = Convolution2D(64, kernel, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

   
