from functools import partial
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
from utils import *

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main(args):
    mnist = input_data.read_data_sets("/tmp/data/")

    n_inputs = 28 * 28
    n_hidden1 = 500
    n_hidden2 = 30
    n_hidden3 = n_hidden1
    n_outputs = n_inputs

    learning_rate = 0.001

    initializer = tf.contrib.layers.variance_scaling_initializer()
    my_dense_layer = partial(
        tf.layers.dense,
        activation=tf.nn.relu,
        kernel_initializer=initializer
    )

    X = tf.placeholder(tf.float32, [None, n_inputs])
    hidden1 = my_dense_layer(X, n_hidden1)
    hidden2_mean = my_dense_layer(hidden1, n_hidden2, activation=None)
    hidden2_gamma = my_dense_layer(hidden1, n_hidden2, activation=None)
    noise = tf.random_normal(tf.shape(hidden2_gamma), dtype=tf.float32)
    hidden2 = hidden2_mean + tf.exp(0.5 * hidden2_gamma) * noise
    hidden3 = my_dense_layer(hidden2, n_hidden3)
    logits = my_dense_layer(hidden3, n_outputs, activation=None)
    outputs = tf.sigmoid(logits)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
    reconstruction_loss = tf.reduce_sum(xentropy)
    latent_loss = 0.5 * tf.reduce_sum(tf.exp(hidden2_gamma) + tf.square(hidden2_mean) - 1 - hidden2_gamma)
    loss = reconstruction_loss + latent_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        for epoch in range(args.n_epochs):
            n_batches = mnist.train.num_examples // args.batch_size
            for iteration in tqdm(range(n_batches)):
                X_batch, y_batch = mnist.train.next_batch(args.batch_size)
                sess.run(training_op, feed_dict={X: X_batch})
            loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss], feed_dict={X: X_batch}) 
            print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)  

        codings_rnd = np.random.normal(size=[args.n_digits, n_hidden2])
        outputs_val = outputs.eval(feed_dict={hidden2: codings_rnd})

    plt.figure(figsize=(8,50))
    for iteration in range(args.n_digits):
        plt.subplot(args.n_digits, 10, iteration + 1)
        plot_image(outputs_val[iteration])


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, help='Number of epochs', default=50)
    parser.add_argument('--batch_size', type=int, help='Number of batch size', default= 150)
    parser.add_argument('--n_digits', type=int, help='Number of digits', default=60)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
