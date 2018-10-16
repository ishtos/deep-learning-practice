import os
import sys
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

import keras
import keras.backend as K
from keras import layers


K.set_image_dim_ordering('th')

latent_dim = 100
height = 28
width = 28
channels = 1


def build_generator(latent_dim):    
    latent_input = layers.Input(shape=(latent_dim, ))
    class_input = layers.Input(shape=(1,), dtype='int32')

    emb = layers.Embedding(10, latent_dim, embeddings_initializer='glorot_normal')(class_input)
    emb = layers.Flatten()(emb)

    generator_input = layers.Multiply()([latent_input, emb])

    x = layers.Dense(1024)(generator_input)
    x = layers.ReLU()(x)
    x = layers.Dense(128 * 7 * 7)(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((128, 7, 7))(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(256, 5, padding='same', bias_initializer='glorot_normal')(x)
    x = layers.ReLU()(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, 5, padding='same', bias_initializer='glorot_normal')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(1, 2, padding='same', bias_initializer='glorot_normal')(x)
    fake_image = layers.Activation('tanh')(x)

    generator = keras.models.Model(input=[latent_input, class_input], output=fake_image)
    
    return generator


def build_discriminator():
    discriminator_input = layers.Input(shape=(channels, height, width))

    x = layers.Conv2D(32, 3, padding='same', strides=(1, 1))(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
   
    x = layers.Conv2D(64, 3, padding='same', strides=(1, 1))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 3, padding='same', strides=(2, 2))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 3, padding='same', strides=(1, 1))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
  
    x = layers.Flatten()(x)

    fake = layers.Dense(1, activation='sigmoid', name='generation')(x)
    aux = layers.Dense(10, activation='softmax', name='auxiliary')(x)

    return keras.models.Model(input=discriminator_input, output=[fake, aux])


def build_combined(latent_dim, generator, discriminator):
    latent_input = layers.Input(shape=(latent_dim, ))
    class_input = layers.Input(shape=(1,), dtype='int32')

    fake_image = generator([latent_input, class_input])

    fake, aux = discriminator(fake_image)
    combined = keras.models.Model(input=[latent_input, class_input], output=[fake, aux])

    return combined


def main(args):
    epochs = args.epochs
    batch_size = args.batch_size
    latent_dim = args.latent_dim
    save_dir = args.save_dir

    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    generator = build_generator(latent_dim)
    generator.compile(
        optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
        loss='binary_crossentropy'
    )

    discriminator.trainable = False
    combined = build_combined(latent_dim, generator, discriminator)
    combined.compile(
        optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=1)

    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=1)

    nb_train, nb_test = x_train.shape[0], x_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(epochs):
        print('Epoch {} of {}'.format(epoch + 1, epochs))

        nb_batches = int(x_train.shape[0] / batch_size)
        progress_bar = keras.utils.generic_utils.Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            progress_bar.update(index)

            noise = np.random.normal(size=(batch_size, latent_dim))
            # noise = np.random.uniform(-1, 1, (batch_size, latent_dim))

            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            sampled_labels = np.random.randint(0, 10, batch_size)

            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            x = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            epoch_disc_loss.append(discriminator.train_on_batch(x, [y, aux_y]))

            noise = np.random.normal(size=(2 * batch_size, latent_dim))
            # noise = np.random.uniform(-1, 1, (2 * batch_size, latent_dim))
            sampled_labels = np.random.randint(0, 10, 2 * batch_size)

            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        noise = np.random.normal(size=(nb_test, latent_dim))
        # noise = np.random.uniform(-1, 1, (nb_test, latent_dim))

        sampled_labels = np.random.randint(0, 10, nb_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        x = np.concatenate((x_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        discriminator_test_loss = discriminator.evaluate(
            x, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        noise = np.random.normal(size=(2 * nb_train, latent_dim))
        # noise = np.random.uniform(-1, 1, (2 * nb_test, latent_dim))
        sampled_labels = np.random.randint(0, 10, 2 * nb_test)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        generator.save_weights(
            os.path.join(save_dir, 'params_generator_epoch_{0:03d}.hdf5'.format(epoch)), True)
        discriminator.save_weights(
            os.path.join(save_dir, 'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch)), True)

        noise = np.random.normal(size=(100, latent_dim))
        # noise = np.random.uniform(-1, 1, (100, latent_dim))

        sampled_labels = np.array([[i] * 10 for i in range(10)]).reshape(-1, 1)

        generated_images = generator.predict([noise, sampled_labels], verbose=0)

        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            os.path.join(save_dir, 'plot_epoch_{0:03d}_generated.png'.format(epoch)))

    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./logs')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
