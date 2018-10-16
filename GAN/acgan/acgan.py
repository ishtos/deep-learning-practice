import os
import numpy as np
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from collections import defaultdict
from tqdm import tqdm
from build import build_discriminator, build_generator, build_gan
from load_data import load_fashion_mnist
import argparse
from PIL import Image


np.random.seed(2018)
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--latent_size', type=int, default=100)
args = parser.parse_args()
CLASS_NUM = 10
LOG_PATH = './log'


def process_discriminator(generator, discriminator, images, labels, batch_size, latent_size, is_train=True):
    noise = np.random.normal(
        loc=0.0, scale=0.5, size=(batch_size, latent_size))
    sampled_labels = np.random.randint(0, 10, batch_size)
    sampled_labels = to_categorical(sampled_labels, num_classes=CLASS_NUM)

    generated_images = generator.predict(
        [noise, sampled_labels], verbose=False)

    X = np.concatenate((images, generated_images))
    y = np.array([1] * batch_size + [0] * batch_size)
    aux_y = np.concatenate((labels, sampled_labels), axis=0)

    if is_train:
        loss = discriminator.train_on_batch(X, [y, aux_y])
    else:
        loss = discriminator.evaluate(X, [y, aux_y], verbose=False)

    return loss


def process_generator(gan, batch_size, latent_size, is_train=True):
    noise = np.random.normal(
        loc=0.0, scale=0.5, size=(2 * batch_size, latent_size))
    sampled_labels = np.random.randint(0, 10, 2 * batch_size)
    sampled_labels = to_categorical(sampled_labels, num_classes=CLASS_NUM)

    trick = np.ones(2 * batch_size)

    if is_train:
        loss = gan.train_on_batch([noise, sampled_labels],
                                  [trick, sampled_labels])
    else:
        loss = gan.evaluate([noise, sampled_labels],
                            [trick, sampled_labels], verbose=False)
    return loss


def generate_100images(generator, latent_size):
    noise = np.random.normal(
        loc=0.0, scale=0.5, size=(100, latent_size))
    sampled_labels = np.array([[i] * 10 for i in range(10)]).flatten()
    sampled_labels = to_categorical(sampled_labels, num_classes=CLASS_NUM)

    generated_images = generator.predict(
        [noise, sampled_labels], verbose=False)

    # arrange them into a grid
    img = (np.concatenate([r.reshape(-1, 28)
                           for r in np.split(generated_images, 10)
                           ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
    img = np.clip(img, 0, 255)

    return img


def main(args):
    epochs = args.epochs
    batch_size = args.batch_size
    latent_size = args.latent_size
    label_size = CLASS_NUM

    lr = 0.0002
    beta_1 = 0.5

    # build the discriminator
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=lr, beta_1=beta_1),
        loss=['binary_crossentropy', 'categorical_crossentropy']
    )
    discriminator.summary()

    # build the generator
    generator = build_generator(latent_size=latent_size, label_size=label_size)
    generator.compile(optimizer=Adam(lr=lr, beta_1=beta_1),
                      loss='binary_crossentropy')
    generator.summary()

    # build the gan
    gan = build_gan(generator, discriminator,
                    latent_size=latent_size, label_size=label_size)
    gan.compile(
        optimizer=Adam(lr=lr, beta_1=beta_1),
        loss=['binary_crossentropy', 'categorical_crossentropy']
    )
    gan.summary()

    # fashion_mnist, shape (..., 28, 28, 1) with range [-1, 1]
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    num_train, num_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(epochs):
        print('Epoch {} of {}'.format(epoch + 1, epochs))

        num_batches = int(X_train.shape[0] / batch_size)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in tqdm(range(num_batches)):
            # train discriminator
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            epoch_disc_loss.append(process_discriminator(
                generator, discriminator,
                images=image_batch, labels=label_batch,
                batch_size=batch_size, latent_size=latent_size,
                is_train=True
            ))

            # train generator
            epoch_gen_loss.append(process_generator(
                gan, batch_size=batch_size, latent_size=latent_size,
                is_train=True
            ))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # evaluate discriminator
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        discriminator_test_loss = process_discriminator(
            generator, discriminator,
            images=X_test, labels=y_test,
            batch_size=num_test, latent_size=latent_size,
            is_train=False
        )

        # evaluate generator
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        generator_test_loss = process_generator(
            gan, batch_size=num_test, latent_size=latent_size,
            is_train=False
        )

        # generate an epoch report on performance
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

        # save weights every epoch
        generator.save_weights(os.path.join(
            LOG_PATH, 'params_generator_epoch_{0:03d}.hdf5'.format(epoch)), True)
        discriminator.save_weights(os.path.join(
            LOG_PATH, 'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch)), True)

        # generate some digits to display
        img = generate_100images(generator, latent_size)
        img_path = os.path.join(
            LOG_PATH, 'plot_epoch_{0:03d}_generated.png'.format(epoch))
        Image.fromarray(img).save(img_path)


if __name__ == '__main__':
    os.makedirs(LOG_PATH, exist_ok=True)
    main(args)
