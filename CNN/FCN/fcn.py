from functools import partial

from keras.layers import Input, Concatenate, Add
from keras.layers.core import Activation
from keras.activations import softmax
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Cropping2D
from keras.models import Model
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input


class FCN:

    def __init__(self, input_shape=(224, 224, 3), fcn_classes=21):
        self.input_shape = input_shape
        self.fcn_classes = fcn_classes
        self.vgg16 = VGG16(include_top=False,
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=input_shape)


    def build_graph(self):
        inputs = Input(shape=self.input_shape)

        h = self.vgg16.layers[1](inputs)
        for i in range(2,11):
            h = self.vgg16.layers[i](h)

        p3 = h
        for i in range(11, 15):
            h = self.vgg16.layers[i](h)
        
        p4 = h
        for i in range(15, 19):
            h = self.vgg16.layers[i](h)
        
        p5 = h

        p3 = Conv2D(self.fcn_classes, kernel_size=1, strides=1,
                    activation='relu', padding='valid')(p3)

        p4 = Conv2D(self.fcn_classes, kernel_size=1,
                    strides=1, activation='relu')(p4)
        p4 = Conv2DTranspose(self.fcn_classes, kernel_size=4,
                             strides=2, padding='valid')(p4)
        p4 = Cropping2D(((1, 1), (1, 1)))(p4)

        p5 = Conv2D(self.fcn_classes, kernel_size=1,
                    strides=1, activation='relu')(p5)
        p5 = Conv2DTranspose(self.fcn_classes, kernel_size=8, 
                             strides=4, padding='valid')(p5)
        p5 = Cropping2D(((2, 2), (2, 2)))(p5)

        h = Add()([p3, p4, p5])
        h = Conv2DTranspose(self.fcn_classes, kernel_size=16,
                            strides=8, padding='valid')(h)
        h = Cropping2D(((4, 4), (4, 4)))(h)

        outputs = Activation(softmax)(h)

        model = Model(inputs=inputs, outputs=outputs, name="FCN")
        model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

        return model
