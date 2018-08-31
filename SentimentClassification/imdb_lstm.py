import os
import sys
import argparse
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, DEBUG
from keras.models import Model
from keras.layers import Input
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.callbacks import EarlyStopping, TensorBoard


logger = getLogger(__name__)
handler = FileHandler(os.path.join('logs', datetime.now().strftime('%Y.%m.%d')+'.log'))
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)


def main(args):
    logger.debug('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=args.max_features)
    logger.debug('train sequences: {}'.format(len(x_train)))
    logger.debug('test sequences: {}'.format(len(x_test)))

    logger.debug('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=args.max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=args.max_len)
    
    logger.debug('x_train shape: {}'.format(x_train.shape))
    logger.debug('x_test shape: {}'.format(x_test.shape))

    logger.debug('Build model ...')

    model = build_graph(args.max_features)

    logger.debug('Train ...')

    earlystopping = EarlyStopping(monitor='val_acc', patience=5)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=32, batch_size=32, 
                              write_graph=False, write_grads=False, write_images=False)

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epoch_size,
              validation_data=(x_test, y_test),
              callbacks=[earlystopping, tensorboard])

    score, acc = model.evaluate(x_test, y_test,
                                batch_size=args.batch_size)

    logger.debug('Test score: {}'.format(score))
    logger.debug('Test accuracy {}'.format(acc))
  

def build_graph(max_features):
    inputs = Input(shape=(80,))

    emb = Embedding(max_features, 128)(inputs)
    lstm = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(emb)
    
    outputs = Dense(1, activation='sigmoid')(lstm)

    model = Model(inputs=inputs, outputs=outputs, name='NSC')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.summary()
    
    return model
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_features', type=int, default=20000)
    parser.add_argument('--max_len', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch_size', type=int, default=15)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--log', type=bool, default=False)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
