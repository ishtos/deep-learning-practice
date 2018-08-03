from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization


class Eyes:

    def __init__(self):
        pass

    def build_graph(self, num_classes=5):
        inputs = Input(shape=(None, 64, 64, 3))

        l_1 = ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(inputs)
        
        l_2 = BatchNormalization()(l_1)
        
        l_3 = ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(l_2)

        l_4 = BatchNormalization()(l_3)

        l_5 = ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True)(l_4)
     
        l_6 = BatchNormalization()(l_5)

        lstm_outputs = ConvLSTM2D(filters=40, kernel_size=(3, 3),
                                 padding='same', return_sequences=False, name='lstm_outputs')(l_6)


        l_8 = BatchNormalization()(lstm_outputs)

        l_9 = Conv2D(32, kernel_size=(3, 3),
                      activation='relu')(l_8)

        l_10 = Conv2D(32, kernel_size=(3, 3),
                      activation='relu')(l_9)

        l_11 = MaxPooling2D(pool_size=(2, 2))(l_10)

        l_12 = Dropout(0.25)(l_11)

        l_13 = Flatten()(l_12)

        l_14 = Dense(128, activation='relu')(l_13)

        l_15 = Dropout(0.5)(l_14)

        cnn_outputs = Dense(num_classes, activation='softmax', name='cnn_outputs')(l_15)

        model = Model(inputs=inputs, outputs=[lstm_outputs, cnn_outputs], name='Eyes')
        model.compile(optimizer='adadelta',
                      loss={'lstm_outputs': 'binary_crossentropy', 'cnn_outputs': 'categorical_crossentropy'},
                      loss_weights={'lstm_outputs': 0.8, 'cnn_outputs': 1.0})

        return model
