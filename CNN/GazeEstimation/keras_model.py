from keras import layers
from keras.applications import VGG16, VGG19, ResNet50
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import Constant, RandomNormal
from keras.wrappers.scikit_learn import KerasRegressor

def build_graph():
    inputs = layers.Input(shape=(448, 448, 3))

    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    # resnet = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    
    output = layers.ReLU()(resnet.output)
    output = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(output)
    
    u = output

    w = output
    w = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), bias_initializer=Constant(value=0.1))(w)
    w = layers.ReLU()(w)
    w = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), bias_initializer=Constant(value=0.1))(w)
    w = layers.ReLU()(w)
    w = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001), bias_initializer=Constant(value=1))(w)

    x = layers.Multiply()([u, w])
    x = layers.ReLU()(x)  
    x = layers.Dropout(0.5)(x)  
    x = layers.Flatten()(x)
    x = layers.Dense(4096)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)  
    x = layers.Dense(4096)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)  
    outputs = layers.Dense(2)(x)

    model = Model(input=inputs, output=outputs)
    model.compile(optimizer='adam', loss='mean_absolute_error')

    return model