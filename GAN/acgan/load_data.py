import numpy as np
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical


def load_fashion_mnist():
    print('Load fashion_mnist...')

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=3)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('Done.')
    print('Train: {}, Test: {}'.format(X_train.shape, X_test.shape))

    return (X_train, y_train), (X_test, y_test)
