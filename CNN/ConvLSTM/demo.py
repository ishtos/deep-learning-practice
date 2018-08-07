import sys
import os 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gc

from keras.callbacks import EarlyStopping
from keras.utils import plot_model

from eyes import Eyes

path = "mnist_test_seq.npy"


def make_datasets(path, train_size, test_size):
    data = np.load(path)
    print("data shape:", data.shape)

    train_x = data[0:19]
    train_y = data[-1]

    train_x = np.transpose(train_x, (1, 0, 2, 3))

    train_x = np.expand_dims(train_x, axis=4)
    train_y = np.expand_dims(train_y, axis=3)

    print("train_x shape:", train_x[0:train_size].shape)
    print("train_y shape:", train_y[0:train_size].shape)
    print("test shape:", train_x[0:test_size].shape)

    return train_x[0:train_size], train_y[0:train_size], train_x[0:test_size]


def imshow(images):
    images = images.reshape(9, 64, 64)
    for i, img in enumerate(images):
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
    
    plt.show()


def main(args):
    train_x, train_y, test = make_datasets(path, args.train_size, args.test_size)

    eyes = Eyes()
    model = eyes.build_graph()
    model.summary()
    plot_model(model, to_file='model.png')

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=0, verbose=1, mode='auto')
    history = model.fit(x=train_x, y=train_y, epochs=args.epochs,
                        verbose=1, batch_size=args.batch_size,
                        validation_split=0.5, callbacks=[earlystop])
    model.save_weights(args.save_weights)
    
    result = model.predict(test[:10])
    imshow(result)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_size', type=int, default=5000)
    parser.add_argument('--test_size', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_weights', type=str, default='my_model_weights.h5')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
