import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import keras
from keras import layers
from keras.datasets import mnist
from keras.preprocessing import image


latent_dim = 100
height = 28
width = 28
channels = 1
CLASS_NUM = 10

def build_generator():
    generator_input = keras.Input(shape=(latent_dim + CLASS_NUM,))

    x = layers.Dense(1024)(generator_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(7 * 7 * 128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Reshape((7, 7, 128))(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(1, 5, padding='same')(x)
    x = layers.Activation('tanh')(x)
    
    generator = keras.models.Model(generator_input, x)
    # generator.summary()

    return generator


def build_discriminator():
    discriminator_input = layers.Input(shape=(height, width, channels+CLASS_NUM))

    x = layers.Conv2D(64, 5, padding='same')(discriminator_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1)(x)
    x = layers.Activation('sigmoid')(x)

    discriminator = keras.models.Model(discriminator_input, x)
    # discriminator.summary()

    return discriminator


def build_gan():
    generator = build_generator()
    discriminator = build_discriminator()

    gan_input = keras.Input(shape=(latent_dim,))
    gan_label_input = keras.Input(shape=(CLASS_NUM,))
    gan_concat_input = layers.Concatenate(axis=-1)([gan_input, gan_label_input])
    gan_generator_output = generator(gan_concat_input)
    gan_image_input = keras.Input(shape=(height, width, CLASS_NUM))
    gan_concat_image_input = layers.Concatenate(axis=3)([gan_generator_output, gan_image_input])

    discriminator.trainable = False
    gan_discriminator_output = discriminator(gan_concat_image_input)
    gan = keras.models.Model([gan_input, gan_label_input, gan_image_input], gan_discriminator_output)

    return gan


def label2image(label):
    images = np.zeros((height, width, CLASS_NUM))
    images[:, :, label] += 1
    
    return images

def main(args):
    (x_train, y_train), (_, _) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], height, width, channels).astype('float32') / 255.

    iterations = args.iterations
    batch_size = args.batch_size

    save_dir = args.save_dir

    generator = build_generator()
    discriminator = build_discriminator()
    discriminator_optimizer = keras.optimizers.Adam(lr=0.0008, decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
    
    gan = build_gan()
    gan_optimizer = keras.optimizers.Adam(lr=0.0004, decay=1e-8)
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

    start = 0
    for step in tqdm(range(iterations)):
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        random_labels = np.random.randint(0, CLASS_NUM, batch_size)
        random_labels_vectors = keras.utils.to_categorical(random_labels, CLASS_NUM)
        random_vectors = np.concatenate([random_latent_vectors, random_labels_vectors], axis=1)
        generated_iamges = generator.predict(random_vectors)
        random_labels_images = np.array([label2image(i) for i in random_labels])
        generated_concat_iamges = np.concatenate((generated_iamges, random_labels_images), axis=3)

        stop = start + batch_size
        real_images = x_train[start:stop]
        real_labels = y_train[start:stop]
        real_labels_images = np.array([label2image(i) for i in real_labels])
        real_concat_images = np.concatenate((real_images, real_labels_images), axis=3)

        combined_images = np.concatenate([generated_concat_iamges, real_concat_images])

        labels = np.concatenate([np.ones((batch_size, 1)),
                                 np.zeros((batch_size, 1))])

        labels += 0.05 * np.random.random(labels.shape)

        d_loss = discriminator.train_on_batch(combined_images, labels)

        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        misleading_targets = np.zeros((batch_size, 1))

        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        random_labels = np.random.randint(0, CLASS_NUM, batch_size)
        random_labels_vectors = keras.utils.to_categorical(random_labels, CLASS_NUM)
        a_loss = gan.train_on_batch([random_latent_vectors, random_labels_vectors, random_labels_images], misleading_targets)

        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0

        if step % 1000 == 0 and step != 0:
            gan.save_weights('gan.h5')

            print('discriminator loss at step %s: %s' % (step, d_loss))
            print('adversarial loss at step   %s: %s' % (step, a_loss))

            img = image.array_to_img(generated_iamges[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'generated' + str(step) + '.png'))

            img = image.array_to_img(real_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real' + str(step) + '.png'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='./logs')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
