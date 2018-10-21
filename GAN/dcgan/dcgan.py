import os
import numpy as np
from keras.optimizers import Adam
from collections import defaultdict
from tqdm import tqdm
from build import build_discriminator, build_generator, build_gan
from load_data import load_mnist
import argparse
from PIL import Image


np.random.seed(2018)
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--latent_size', type=int, default=100)
# index 0: mnist
# index 1: fashion_mnist
parser.add_argument('--mnist', type=int, default=0)
args = parser.parse_args()
LOG_PATH = './log'


def process_discriminator(generator, discriminator, images, batch_size, latent_size, is_train=True):
    # as using batchnorm, have to train discriminator as spliting fake and true
    noise = np.random.uniform(-1, 1, size=(batch_size, latent_size))
    generated_images = generator.predict(noise, verbose=False)

    y_true = np.ones(batch_size)
    y_fake = np.zeros(batch_size)

    if is_train:
        loss_true = discriminator.train_on_batch(images, y_true)
        loss_fake = discriminator.train_on_batch(generated_images, y_fake)
        loss = loss_true + loss_fake
    else:
        loss_true = discriminator.evaluate(images, y_true, verbose=False)
        loss_fake = discriminator.evaluate(generated_images, y_fake,
                                           verbose=False)
        loss = loss_true + loss_fake

    return loss


def process_generator(gan, batch_size, latent_size, is_train=True):
    noise = np.random.uniform(-1, 1, size=(2 * batch_size, latent_size))
    trick = np.ones(2 * batch_size)

    if is_train:
        loss = gan.train_on_batch(noise, trick)
    else:
        loss = gan.evaluate(noise, trick, verbose=False)

    return loss


def generate_100images(generator, latent_size):
    noise = np.random.uniform(-1, 1, size=(100, latent_size))
    generated_images = generator.predict(noise, verbose=False)

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
    mnist = args.mnist

    lr = 0.0002
    beta_1 = 0.5

    # build the discriminator
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=lr, beta_1=beta_1),
        loss='binary_crossentropy')
    discriminator.summary()

    # build the generator
    generator = build_generator(latent_size=latent_size)
    generator.compile(
        optimizer=Adam(lr=lr, beta_1=beta_1),
        loss='binary_crossentropy')
    generator.summary()

    # build the gan
    gan = build_gan(generator, discriminator, latent_size=latent_size)
    gan.compile(
        optimizer=Adam(lr=lr, beta_1=beta_1),
        loss='binary_crossentropy')
    gan.summary()

    # shape (..., 28, 28, 1) with range [-1, 1]
    if mnist == 0:
        (X_train, y_train), (X_test, y_test) = load_mnist(is_fashion=False)
    elif mnist == 1:
        (X_train, y_train), (X_test, y_test) = load_mnist(is_fashion=True)

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

            epoch_disc_loss.append(process_discriminator(
                generator, discriminator, images=image_batch,
                batch_size=batch_size, latent_size=latent_size, is_train=True
            ))

            # train generator
            epoch_gen_loss.append(process_generator(
                gan, batch_size=batch_size, latent_size=latent_size, is_train=True
            ))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # evaluate discriminator
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        discriminator_test_loss = process_discriminator(
            generator, discriminator, images=X_test,
            batch_size=num_test, latent_size=latent_size, is_train=False
        )

        # evaluate generator
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        generator_test_loss = process_generator(
            gan, batch_size=num_test, latent_size=latent_size, is_train=False
        )

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 30)

        ROW_FMT = '{0:<22s} | {1:<4.2f}'
        print(ROW_FMT.format('generator (train)',
                             train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             test_history['discriminator'][-1]))

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
