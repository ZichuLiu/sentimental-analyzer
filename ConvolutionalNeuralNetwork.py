import os
import pickle
import mini_batches
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def conv_net(sample, iter_scale=10000, restore=False, checkpoint_file=None):
    with tf.Session() as sess:
        x_dim = (705, 880, 3)
        X = tf.placeholder(tf.float32, (None,) + x_dim)
        Y = tf.placeholder(tf.float32, (None, 3))

        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        # shape of img here: (177,220,64)

        W_fc1 = weight_variable([177 * 220 * 64, 200])
        b_fc1 = bias_variable([200])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 177 * 220 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([200, 3])
        b_fc2 = bias_variable([3])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y = tf.nn.softmax(y_conv)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.initialize_all_variables())

        for i in range(1,iter_scale):
            print("HI")
            training_samples = sample.get_train()
            if i % 100 == 0:
                validation_samples = sample.get_validation()
                train_accuracy = accuracy.eval(feed_dict={X: training_samples[0], Y: training_samples[1]})
                validation_accuracy = accuracy.eval(feed_dict={X: validation_samples[0], Y: validation_samples[1]})
                print(
                    "step %d, training accuracy %g, validation accuracy %g" % (i, train_accuracy, validation_accuracy))
            train_step.run(feed_dict={X: training_samples[0][0:10], Y: training_samples[1][0:10],keep_prob: 0.5})


if __name__ == '__main__':
    path = "C:/Users/v-zicliu/Desktop/icml/naive cnn/0/"
    sample = mini_batches.MiniBatches(path)
    images = sample.get_train()
    conv_net(sample)
