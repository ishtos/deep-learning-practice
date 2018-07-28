import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data


def build_graph(learning_rate=0.01):
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    
    with tf.name_scope("dnn"): 
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2")
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return X, y, training_op, accuracy

def main():
    mnist = input_data.read_data_sets("/tmp/data/")

    X, y, training_op, accuracy = build_graph()

    n_epochs = 10
    batch_size = 50

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for _ in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
            print(epoch+1, "Train accuracy:",acc_train, "Val accuracy:", acc_val)

if __name__ == '__main__':
    main()
