# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
import os

os.chdir('/home/pig/PycharmProjects/Graduate/bishe')
n_input = 2358-208  # MNIST data input (img shape: 28*28)
n_steps = 208  # timesteps
n_hidden = 512  # hidden layer num of features
n_classes = 1  # MNIST total classes (0-9 digits)
params = {
    "learning_rate": 0.01,
    "training_iters": 20000,
    "batch_size": 2,
    "display_step": 10
}


def rnn_model(x, weights, biases):
    """RNN (LSTM or GRU) model for image"""
    # 传进来的x的size为:(128,28,28)128batchs,28个steps,每个step对应28个input
    x = tf.reshape(x, [-1, n_input])
    # 转换成(128*28,28)
    x_in = tf.matmul(x, weights['in']) + biases['in']
    x_in = tf.reshape(x_in, [-1, n_steps, n_hidden])

    # 传入lstm的cell
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # 在lstm中,state分为主线c_state,和辅线m_state
    _init_state = lstm_cell.zero_state(params['batch_size'], dtype=tf.float32)

    outputs, states = rnn.dynamic_rnn(lstm_cell, x_in, initial_state=_init_state, time_major=False)
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))
    # unpack为[(batch,outputs)]*steps
    result = tf.matmul(tf.reshape(outputs, [-1, n_hidden]), weights['out']) + biases['out']
    return result


"""Train an image classifier"""
"""Step 0: load image data and training parameters"""
import data_gen

# mnist = input_data.read_data_sets("./data/", one_hot=True)
# parameter_file = sys.argv[1]#从命令行接受参数
# ================加载数据集==================
from data_gen import *
mnist = dataset(*gen_X_y('bookmarks'))

# import cPickle as pickle
# f=open('book.pkl','r')
# mnist=pickle.load(f)
# f.close()

"""Step 1: build a rnn model for image"""
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden, ])),
    'out': tf.Variable(tf.constant([n_classes, ], dtype=tf.float32))
}

pred = rnn_model(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(params['learning_rate']).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

"""Step 2: train the image classification model"""
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 0
    tf.device("/gpu:0")

    """Step 2.0: create a directory for saving model files"""
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())

    """Step 2.1: train the image classifier batch by batch"""
    while step * params['batch_size'] < params['training_iters']:
        batch_x, batch_y = mnist.next_batch(params['batch_size'])
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((params['batch_size'], n_steps, n_input))
        batch_y=batch_y.reshape(-1,1)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        """Step 2.2: save the model"""
        if step % params['display_step'] == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=step)
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print('Iter: {}, Loss: {:.6f}, Accuracy: {:.6f}'.format(step * params['batch_size'], loss, acc))
        step += 1
    print("The training is done")

    """Step 3: test the model"""
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

print 1