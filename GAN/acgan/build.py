from keras.layers import Input, Dense, Reshape, Flatten, concatenate, Dropout, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Model
import numpy as np


np.random.seed(2018)


def build_generator(latent_size=100, label_size=10):
    # input latent and label
    generator_latent = Input(shape=(latent_size,), name='generator_latent')
    generator_label = Input(shape=(label_size,), name='generator_label')
    generator_input = concatenate([generator_latent, generator_label])

    # reshape 2d
    generator_dense1 = Dense(1024, name='generator_dense1'
                             )(generator_input)
    generator_batchnorm1 = BatchNormalization(name='generator_batchnorm1'
                                              )(generator_dense1)
    generator_relu1 = Activation('relu', name='generator_relu1'
                                 )(generator_batchnorm1)
    generator_dense2 = Dense(7 * 7 * 128, name='generator_dense2'
                             )(generator_relu1)
    generator_batchnorm2 = BatchNormalization(name='generator_batchnorm2'
                                              )(generator_dense2)
    generator_relu2 = Activation('relu', name='generator_relu2'
                                 )(generator_batchnorm2)
    generator_reshape = Reshape(target_shape=(7, 7, 128),
                                name='generator_reshape')(generator_relu2)

    # deconvolution
    generator_deconv1 = Conv2DTranspose(filters=64, kernel_size=5, strides=2,
                                        padding='same', kernel_initializer='glorot_normal',
                                        name='generator_deconv1')(generator_reshape)
    generator_batchnorm3 = BatchNormalization(name='generator_batchnorm3',
                                              )(generator_deconv1)
    generator_relu3 = Activation('relu', name='generator_relu3'
                                 )(generator_batchnorm3)
    generator_deconv2 = Conv2DTranspose(filters=1, kernel_size=5, strides=2,
                                        padding='same', kernel_initializer='glorot_normal',
                                        name='generator_deconv2')(generator_relu3)
    generator_output = Activation('tanh', name='generator_output',
                                  )(generator_deconv2)

    model = Model(input=[generator_latent, generator_label],
                  output=generator_output)
    return model


def build_discriminator():
    discriminator_input = Input(shape=(28, 28, 1))

    # conv
    discriminator_conv1 = Conv2D(filters=64, kernel_size=5, strides=2,
                                 padding='same', kernel_initializer='glorot_normal',
                                 name='discriminator_conv1')(discriminator_input)
    discriminator_batchnorm1 = BatchNormalization(name='discriminator_batchnorm1'
                                                  )(discriminator_conv1)
    discriminator_leakyrelu1 = LeakyReLU(alpha=0.2,
                                         name='discriminator_leakyrelu1')(discriminator_batchnorm1)
    discriminator_dropout1 = Dropout(0.5, name='discriminator_dropout1',
                                     )(discriminator_leakyrelu1)
    discriminator_conv2 = Conv2D(filters=128, kernel_size=5, strides=2,
                                 padding='same', kernel_initializer='glorot_normal',
                                 name='discriminator_conv2')(discriminator_dropout1)
    discriminator_batchnorm2 = BatchNormalization(name='discriminator_batchnorm2',
                                                  )(discriminator_conv2)
    discriminator_leakyrelu2 = LeakyReLU(alpha=0.2,
                                         name='discriminator_leakyrelu2')(discriminator_batchnorm2)
    discriminator_dropout2 = Dropout(0.5, name='discriminator_dropout2',
                                     )(discriminator_leakyrelu2)

    # linear
    discriminator_flatten = Flatten(name='discriminator_flatten'
                                    )(discriminator_dropout2)
    discriminator_dense1 = Dense(256, name='discriminator_dense1'
                                 )(discriminator_flatten)
    discriminator_batchnorm3 = BatchNormalization(name='discriminator_batchnorm3'
                                                  )(discriminator_dense1)
    discriminator_leakyrelu3 = LeakyReLU(alpha=0.2,
                                         name='discriminator_leakyrelu3')(discriminator_dense1)
    discriminator_dropout3 = Dropout(0.5, name='discriminator_dropout3',
                                     )(discriminator_leakyrelu3)
    discriminator_output = Dense(1, activation='sigmoid',
                                 name='discriminator_output')(discriminator_dropout3)
    auxiliary_output = Dense(10, activation='softmax',
                             name='auxiliary_output')(discriminator_dropout3)

    model = Model(input=discriminator_input,
                  output=[discriminator_output, auxiliary_output])
    return model


def build_gan(generator, discriminator, latent_size=100, label_size=10):
    gan_latent = Input(shape=(latent_size,), name='gan_latent')
    gan_label = Input(shape=(label_size,), name='gan_label')
    gan_generator_output = generator([gan_latent, gan_label])

    discriminator.trainable = False
    gan_discriminator_output, gan_auxiliary_output = discriminator(
        gan_generator_output)

    model = Model(input=[gan_latent, gan_label],
                  output=[gan_discriminator_output, gan_auxiliary_output])
    return model
