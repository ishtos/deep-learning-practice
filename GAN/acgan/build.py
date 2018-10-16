from keras.layers import Input, Dense, Reshape, Flatten, concatenate, Dropout, BatchNormalization
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
    generator_dense = Dense(7 * 7 * 128, activation='relu',
                            name='generator_dense')(generator_input)
    generator_reshape = Reshape(target_shape=(7, 7, 128),
                                name='generator_reshape')(generator_dense)

    # deconvolution
    generator_deconv1 = Conv2DTranspose(filters=64, kernel_size=5, strides=2,
                                        padding='same', activation='relu',
                                        kernel_initializer='glorot_normal',
                                        name='generator_deconv1')(generator_reshape)
    generator_batchnorm1 = BatchNormalization(name='generator_batchnorm1',
                                              )(generator_deconv1)
    generator_deconv2 = Conv2DTranspose(filters=32, kernel_size=5, strides=2,
                                        padding='same', activation='relu',
                                        kernel_initializer='glorot_normal',
                                        name='generator_deconv2')(generator_batchnorm1)
    generator_batchnorm2 = BatchNormalization(name='generator_batchnorm2',
                                              )(generator_deconv2)
    generator_output = Conv2DTranspose(filters=1, kernel_size=2, strides=1,
                                       padding='same', activation='tanh',
                                       kernel_initializer='glorot_normal',
                                       name='generator_output')(generator_batchnorm2)

    model = Model(input=[generator_latent, generator_label],
                  output=generator_output)
    return model


def build_discriminator():
    discriminator_input = Input(shape=(28, 28, 1))

    # conv
    discriminator_conv1 = Conv2D(filters=16, kernel_size=3, strides=2,
                                 padding='valid', kernel_initializer='glorot_normal',
                                 name='discriminator_conv1')(discriminator_input)
    discriminator_leakyrelu1 = LeakyReLU(alpha=0.2,
                                         name='discriminator_leakyrelu1')(discriminator_conv1)
    discriminator_dropout1 = Dropout(0.5, name='discriminator_dropout1',
                                     )(discriminator_leakyrelu1)
    discriminator_conv2 = Conv2D(filters=32, kernel_size=3, strides=1,
                                 padding='valid', kernel_initializer='glorot_normal',
                                 name='discriminator_conv2')(discriminator_dropout1)
    discriminator_leakyrelu2 = LeakyReLU(alpha=0.2,
                                         name='discriminator_leakyrelu2')(discriminator_conv2)
    discriminator_batchnorm2 = BatchNormalization(name='discriminator_batchnorm2',
                                                  )(discriminator_leakyrelu2)
    discriminator_dropout2 = Dropout(0.5, name='discriminator_dropout2',
                                     )(discriminator_batchnorm2)
    discriminator_conv3 = Conv2D(filters=64, kernel_size=3, strides=2,
                                 padding='valid', kernel_initializer='glorot_normal',
                                 name='discriminator_conv3')(discriminator_dropout2)
    discriminator_leakyrelu3 = LeakyReLU(alpha=0.2,
                                         name='discriminator_leakyrelu3')(discriminator_conv3)
    discriminator_batchnorm3 = BatchNormalization(name='discriminator_batchnorm3',
                                                  )(discriminator_leakyrelu3)
    discriminator_dropout3 = Dropout(0.5, name='discriminator_dropout3',
                                     )(discriminator_batchnorm3)
    discriminator_conv4 = Conv2D(filters=128, kernel_size=3, strides=1,
                                 padding='valid', kernel_initializer='glorot_normal',
                                 name='discriminator_conv4')(discriminator_dropout3)
    discriminator_leakyrelu4 = LeakyReLU(alpha=0.2,
                                         name='discriminator_leakyrelu4')(discriminator_conv4)
    discriminator_batchnorm4 = BatchNormalization(name='discriminator_batchnorm4',
                                                  )(discriminator_leakyrelu4)
    discriminator_dropout4 = Dropout(0.5, name='discriminator_dropout4',
                                     )(discriminator_batchnorm4)

    # linear
    discriminator_flatten = Flatten(name='discriminator_flatten'
                                    )(discriminator_dropout4)
    discriminator_output = Dense(1, activation='sigmoid',
                                 name='discriminator_output')(discriminator_flatten)
    auxiliary_output = Dense(10, activation='softmax',
                             name='auxiliary_output')(discriminator_flatten)

    model = Model(input=discriminator_input,
                  output=[discriminator_output, auxiliary_output])
    return model


def build_gan(generator, discriminator, latent_size=100, label_size=10):
    gan_latent = Input(shape=(latent_size,), name='gan_latent')
    gan_label = Input(shape=(label_size,), name='gan_label')
    gan_generator_output = generator([gan_latent, gan_label])

    discriminator.trainable = True
    gan_discriminator_output, gan_auxiliary_output = discriminator(
        gan_generator_output)

    model = Model(input=[gan_latent, gan_label],
                  output=[gan_discriminator_output, gan_auxiliary_output])
    return model
