from keras.layers import Input, Dense, Reshape, Flatten, concatenate, Dropout, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Model
import numpy as np


np.random.seed(2018)


def build_generator(latent_size=100):
    # input latent
    generator_latent = Input(shape=(latent_size,), name='generator_latent')

    # reshape 2d
    generator_dense1 = Dense(7 * 7 * 128, name='generator_dense1'
                             )(generator_latent)
    generator_relu1 = Activation('relu', name='generator_relu1'
                                 )(generator_dense1)
    generator_reshape = Reshape(target_shape=(7, 7, 128),
                                name='generator_reshape')(generator_relu1)

    # deconvolution
    generator_deconv1 = Conv2DTranspose(filters=64, kernel_size=5, strides=2,
                                        padding='same', kernel_initializer='glorot_normal',
                                        name='generator_deconv1')(generator_reshape)
    generator_batchnorm1 = BatchNormalization(name='generator_batchnorm1',
                                              )(generator_deconv1)
    generator_relu2 = Activation('relu', name='generator_relu2'
                                 )(generator_batchnorm1)
    generator_deconv2 = Conv2DTranspose(filters=1, kernel_size=5, strides=2,
                                        padding='same', kernel_initializer='glorot_normal',
                                        name='generator_deconv2')(generator_relu2)
    generator_tanh1 = Activation('tanh', name='generator_tanh1',
                                 )(generator_deconv2)

    model = Model(input=generator_latent, output=generator_tanh1)

    return model


def build_discriminator():
    discriminator_image = Input(shape=(28, 28, 1), name='discriminator_image')

    # conv
    discriminator_conv1 = Conv2D(filters=64, kernel_size=5, strides=2,
                                 padding='same', kernel_initializer='glorot_normal',
                                 name='discriminator_conv1')(discriminator_image)
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
    discriminator_dense1 = Dense(1, name='discriminator_dense1'
                                 )(discriminator_flatten)
    discriminator_sigmoid1 = Activation('sigmoid', name='discriminator_sigmoid1'
                                        )(discriminator_dense1)

    model = Model(input=discriminator_image, output=discriminator_sigmoid1)

    return model


def build_gan(generator, discriminator, latent_size=100):
    gan_latent = Input(shape=(latent_size,), name='gan_latent')
    gan_generator_output = generator(gan_latent)

    discriminator.trainable = False
    gan_discriminator_output = discriminator(gan_generator_output)

    model = Model(input=gan_latent, output=gan_discriminator_output)

    return model
