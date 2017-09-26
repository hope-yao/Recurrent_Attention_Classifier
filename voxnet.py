
# Author: Hope Yao
#
# ==============================================================================
"""Contains a variant of the Voxnet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  shape = W.get_shape()
  binary = tf.sign(W) + 1
  norm = tf.norm(tf.norm(W,ord=1,axis=0)/shape[0].value,ord=1,axis=0)/shape[1].value
  W_b = binary*norm*2
  #return tf.nn.conv2d(x, W_b, strides=[1, 1, 1, 1], padding='SAME')
  return tf.nn.conv2d(x, W_b, strides=[1, 1, 1, 1], padding='SAME')



def voxnet(images, num_classes=10):
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

    num_ch = [16, 32]
    # 3D convolution
    filter1 = weight_variable([5, 5, 5, 1, num_ch[0]])
    b_conv1 = bias_variable([num_ch[0]])
    net = tf.nn.relu(tf.nn.conv3d(input=images, filter=filter1, strides=[1,2,2,2,1], padding='SAME')+b_conv1)
    # 3D convolution
    filter2 = weight_variable([5, 5, 5, num_ch[0], num_ch[1]])
    b_conv2 = bias_variable([num_ch[1]])
    net = tf.nn.relu(tf.nn.conv3d(input=net, filter=filter2, strides=[1,2,2,2,1], padding='SAME')+b_conv2)

    # net = tf.nn.max_pool3d(net, 2, 2, name='pool1')
    net = tf.contrib.slim.flatten(net)
    end_points['Flatten'] = net

    mid_output = net = tf.contrib.slim.fully_connected(net, 1024, scope='fc3')
    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
    #                    scope='dropout3')
    logits = tf.contrib.slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc4')

    end_points['Logits'] = logits
    end_points['Predictions'] = tf.nn.softmax(logits)

    return end_points


if __name__ == '__main__':
    from tqdm import tqdm
    import numpy as np
    from utils import *

    batch_size = 128
    n_class = 10
    ch_in = 1
    vox_size = 32
    lr = 0.01

    x = tf.placeholder(tf.float32,shape=(batch_size, vox_size, vox_size, vox_size, ch_in))
    y = tf.placeholder(tf.float32,shape=(batch_size, n_class))
    end_points = voxnet(x)


    ## LOSS FUNCTION ##
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=end_points['Logits']))
    err = tf.reduce_mean(tf.abs(y-end_points['Predictions']))

    ## OPTIMIZER ##
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdadeltaOptimizer(learning_rate)
    grads=optimizer.compute_gradients(cost)
    # for i,(g,v) in enumerate(grads):
    #     if g is not None:
    #         grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
    train_op=optimizer.apply_gradients(grads)


    ## Monitor ##
    # saver = tf.train.Saver() # saves variables learned during training
    logdir, modeldir = creat_dir("voxnet_lr{}".format(lr))
    saver = tf.train.Saver()
    #saver.restore(sess, "*.ckpt")
    summary_writer = tf.summary.FileWriter(logdir)
    summary_op = tf.summary.merge([
        tf.summary.scalar("loss/loss", cost),
        tf.summary.scalar("err/err_last", err),
        tf.summary.scalar("lr/lr", learning_rate),
    ])

    ## get data ##
    from fuel.datasets.hdf5 import H5PYDataset
    datafile_hdf5 = './data/ModelNet/ModelNet10.hdf5'
    train_set = H5PYDataset(datafile_hdf5, which_sets=('train',))
    handle = train_set.open()
    test_set = H5PYDataset(datafile_hdf5, which_sets=('test',))
    handle = test_set.open()
    train_data = train_set.get_data(handle, slice(0, train_set.num_examples))
    test_data = test_set.get_data(handle, slice(0, test_set.num_examples))

    x_train = train_data[0].reshape((train_data[0].shape[0],32,32,32,1))
    y_train = np.zeros((train_data[1].shape[0], 10))
    y_train[np.arange(train_data[1].shape[0]), train_data[1][:, 0] - 1] = 1
    x_test = test_data[0].reshape((test_data[0].shape[0],32,32,32,1))
    y_test = np.zeros((test_data[1].shape[0], 10))
    y_test[np.arange(test_data[1].shape[0]), test_data[1][:, 0] - 1] = 1

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

    count = 0
    it_per_ep = int( len(x_train) / batch_size )
    for itr in range(5000):
        for i in tqdm(range(it_per_ep)):
            x_input = x_train[i*batch_size : (i + 1)*batch_size]
            y_input = y_train[i*batch_size : (i + 1)*batch_size]
            feed_dict_train = {x: x_input, y: y_input}
            results = sess.run(train_op, feed_dict_train)

            if count % 100 == 0:
                rand_idx = np.random.random_integers(0,len(x_test)-1,size=batch_size)
                x_input = x_test[rand_idx]
                y_input = y_test[rand_idx]
                feed_dict_test = {x: x_input, y: y_input}
                train_result = sess.run([cost, err], feed_dict_train)
                test_result = sess.run([cost, err], feed_dict_test)
                print("iter=%d : train_cost: %f train_err_last: %f test_cost: %f test_err_last: %f" %
                      (count, train_result[0], train_result[-1], test_result[0], test_result[-1]))
                summary = sess.run(summary_op, feed_dict_train)
                summary_writer.add_summary(summary, itr)
                summary_writer.flush()

            # if itr % 1000 == 0:
            #     sess.run(tf.assign(learning_rate, learning_rate * 0.5))

            if count % 10000 == 0:
                snapshot_name = "%s_%s" % ('experiment', str(count))
                fn = saver.save(sess, "%s/%s.ckpt" % (modeldir, snapshot_name))
                print("Model saved in file: %s" % fn)
            count += 1
