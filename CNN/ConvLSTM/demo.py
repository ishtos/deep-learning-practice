from eyes import Eyes
from keras.utils import plot_model


def main():
    eyes = Eyes()
    model = eyes.build_graph()

    model.summary()
    plot_model(model, to_file='model.png')


if __name__ == '__main__':
    main()
