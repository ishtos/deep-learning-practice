import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import keras
from keras import layers
from keras.datasets import mnist
from keras.preprocessing import image


latent_dim = 28
height = 28
width = 28
channels = 1


def build_generator():
    generator_input = keras.Input(shape=(latent_dim,))

    x = layers.Dense(128 * 14 * 14)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((14, 14, 128))(x)

    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(channels, 7,activation='tanh', padding='same')(x)

    generator = keras.models.Model(generator_input, x)
    # generator.summary()

    return generator


def build_discriminator():
    discriminator_input = layers.Input(shape=(height, width, channels))

    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(1, activation='sigmoid')(x)

    discriminator = keras.models.Model(discriminator_input, x)
    # discriminator.summary() 

    discriminator_optimizer = keras.optimizers.RMSprop(
        lr=0.0008, clipvalue=1.0, decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer,
                          loss='binary_crossentropy')

    return discriminator


def build_gan():
    generator = build_generator()
    discriminator = build_discriminator()

    discriminator.trainable = False

    gan_input = keras.Input(shape=(latent_dim, ))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    gan_optimizer = keras.optimizers.RMSprop(
        lr=0.0004, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=gan_optimizer, 
                loss='binary_crossentropy')

    return gan


def main(args):
    (x_train, y_train), (_, _) = mnist.load_data()

    x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

    iterations = args.iterations
    batch_size = args.batch_size

    save_dir = args.save_dir

    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan()

    start = 0
    for step in tqdm(range(iterations)):
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        generated_iamges = generator.predict(random_latent_vectors)

        stop = start + batch_size
        real_images = x_train[start:stop]
        combined_images = np.concatenate([generated_iamges, real_images])

        labels = np.concatenate([np.ones((batch_size, 1)),
                                 np.zeros((batch_size, 1))])

        labels += 0.05 * np.random.random(labels.shape)

        d_loss = discriminator.train_on_batch(combined_images, labels)

        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        misleading_targets = np.zeros((batch_size, 1))

        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0
        
        if step % 1000 == 0:
            gan.save_weights('gan.h5')

            print('discriminator loss at step %s: %s' % (step, d_loss))
            print('adversarial loss at step   %s: %s' % (step, a_loss))

            img = image.array_to_img(generated_iamges[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'generated' + str(step) + '.png'))

            img = image.array_to_img(real_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real' + str(step) + '.png'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='./logs')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

