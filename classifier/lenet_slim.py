# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a variant of the LeNet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils.utils import creat_dir
from tensorflow.examples.tutorials import mnist
import numpy as np
slim = tf.contrib.slim
import os


def lenet(images, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet'):
    """Creates a variant of the LeNet model.

    Note that since the output is a set of 'logits', the values fall in the
    interval of (-infinity, infinity). Consequently, to convert the outputs to a
    probability distribution over the characters, one will need to convert them
    using the softmax function:

        logits = lenet.lenet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

    Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

    Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
    """
    end_points = {}

    # with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
    net = slim.conv2d(images, 32, 5, scope='conv1')
    net = slim.max_pool2d(net, 2, 2, scope='pool1')
    net = slim.conv2d(net, 64, 5, scope='conv2')
    net = slim.max_pool2d(net, 2, 2, scope='pool2')
    net = slim.flatten(net)
    end_points['Flatten'] = net

    mid_output = net = slim.fully_connected(net, 1024, scope='fc3')
    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
    #                    scope='dropout3')
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc4')

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return mid_output, end_points

lenet.default_image_size = 28


def lenet_arg_scope(weight_decay=0.0):
  """Defines the default lenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
      activation_fn=tf.nn.relu) as sc:
    return sc


def main(cfg):
    img_size = cfg['img_size']
    img_dim = cfg['img_dim']
    batch_size = cfg['batch_size']
    num_glimpse = cfg['num_glimpse']
    glimpse_size = cfg['glimpse_size']
    data_path = cfg['data_path']
    lr = cfg['lr']
    n_class = cfg['n_class']

    x = tf.placeholder(tf.float32,shape=(batch_size, img_size, img_size,1))
    y = tf.placeholder(tf.float32,shape=(batch_size, n_class))
    mid_output, end_points = lenet(x, num_classes=10, is_training=True,
                                   dropout_keep_prob=0.99,
                                   scope='LeNet')
    ## LOSS FUNCTION ##
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=end_points['Logits']))

    ## OPTIMIZER ##
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    grads=optimizer.compute_gradients(cost)
    # for i,(g,v) in enumerate(grads):
    #     if g is not None:
    #         grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
    train_op=optimizer.apply_gradients(grads)


    ## Monitor ##
    overall_acc, overall_acc_update = tf.metrics.accuracy(labels=tf.argmax(y,1), predictions=tf.argmax(end_points['Predictions'],1))
    overall_err = 1.-overall_acc
    overall_err_update = overall_acc_update

    # saver = tf.train.Saver() # saves variables learned during training
    logdir, modeldir = creat_dir("LeNet")
    saver = tf.train.Saver()
    #saver.restore(sess, "*.ckpt")
    summary_writer = tf.summary.FileWriter(logdir)
    summary_op_train = tf.summary.merge([
        tf.summary.scalar("loss/loss_train", cost),
        tf.summary.scalar("loss/err_last_train", overall_err),
        tf.summary.scalar("lr/lr", learning_rate),
    ])

    summary_op_test = tf.summary.merge([
        tf.summary.scalar("loss/loss_test", cost),
        tf.summary.scalar("loss/err_last_test", overall_err),
    ])

    ## preparing data #
    data_directory = os.path.join(data_path)
    if not os.path.exists(data_directory ):
        os.makedirs(data_directory)
    train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data
    test_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).test # binarized (0-1) mnist data

    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)
    train_iters=500000
    for itr in range(train_iters):
        x_train,y_train = train_data.next_batch(batch_size) # xtrain is (batch_size x img_size)
        x_train = np.reshape(x_train,(x_train.shape[0],img_size,img_size,1))
        feed_dict_train={x:x_train, y:y_train}
        results = sess.run(train_op,feed_dict_train)

        if itr%100==0:
            sess.run(tf.local_variables_initializer())
            for ii in range(100):
                x_test, y_test = test_data.next_batch(batch_size)  # xtrain is (batch_size x img_size)
                feed_dict_test = {x: np.reshape(x_test, (x_test.shape[0], img_size, img_size, 1)), y: y_test}
                sess.run(overall_err_update, feed_dict_test)
            summary_test = sess.run(summary_op_test, feed_dict_test)
            summary_writer.add_summary(summary_test, itr)
            for ii in range(100):
                x_train, y_train = train_data.next_batch(batch_size)  # xtrain is (batch_size x img_size)
                feed_dict_train = {x: np.reshape(x_test, (x_test.shape[0], img_size, img_size, 1)), y: y_test}
                sess.run(overall_err_update, feed_dict_train)
            summary_train = sess.run(summary_op_train, feed_dict_train)
            summary_writer.add_summary(summary_train, itr)
            summary_writer.flush()

        if itr%10000==0:
            snapshot_name = "%s_%s" % ('experiment', str(itr))
            fn = saver.save(sess, "%s/%s.ckpt" % (modeldir, snapshot_name))
            print("Model saved in file: %s" % fn)


if __name__ == '__main__':
    cfg = {'batch_size': 128,
           'img_dim': 2,
           'img_size': 28,
           'n_class': 10,
           'num_glimpse': 5,
           'glimpse_size': 3,
           'data_path': './data/mnist',
           'lr': 1e-3
           }
    main(cfg)