
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



def voxnet(images, shared_conv_var, num_classes=10, is_training=False):
    conv1_w = shared_conv_var['conv1_w']
    conv1_b = shared_conv_var['conv1_b']
    conv2_w = shared_conv_var['conv2_w']
    conv2_b = shared_conv_var['conv2_b']

    end_points = {}
    # 3D convolution
    net = tf.nn.elu(tf.nn.conv3d(input=images, filter=conv1_w, strides=[1,2,2,2,1], padding='SAME') + conv1_b)
    net = tf.nn.elu(tf.nn.conv3d(input=net, filter=conv2_w, strides=[1,2,2,2,1], padding='SAME') + conv2_b)

    # net = tf.nn.max_pool3d(net, 2, 2, name='pool1')
    net = tf.contrib.slim.flatten(net)
    end_points['Flatten'] = net

    mid_output = net = tf.contrib.slim.fully_connected(net, 128, scope='fc3')
    # net = tf.contrib.slim.dropout(net, 0.5, is_training=is_training,
    #                    scope='dropout3')
    logits = tf.contrib.slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc4')

    end_points['Logits'] = logits
    end_points['Predictions'] = tf.nn.softmax(logits)

    return end_points


if __name__ == '__main__':
    from tqdm import tqdm
    import numpy as np
    from utils.utils import *
    from shutil import copyfile


    batch_size = 128
    n_class = 10
    ch_in = 1
    vox_size = 32
    lr = 0.001

    x = tf.placeholder(tf.float32,shape=(batch_size, vox_size, vox_size, vox_size, ch_in))
    y = tf.placeholder(tf.float32,shape=(batch_size, n_class))
    is_training = tf.placeholder(tf.bool,())

    num_ch = [16, 32, 64]
    conv1_w = weight_variable([5, 5, 5, 1, num_ch[0]])
    conv1_b = bias_variable([num_ch[0]])
    conv11_w = weight_variable([5, 5, 5, num_ch[0], num_ch[0]])
    conv11_b = bias_variable([num_ch[0]])
    conv2_w = weight_variable([5, 5, 5, num_ch[0], num_ch[1]])
    conv2_b = bias_variable([num_ch[1]])
    conv21_w = weight_variable([5, 5, 5, num_ch[1], num_ch[1]])
    conv21_b = bias_variable([num_ch[1]])
    conv3_w = weight_variable([5, 5, 5, num_ch[1], num_ch[2]])
    conv3_b = bias_variable([num_ch[2]])
    conv31_w = weight_variable([5, 5, 5, num_ch[2], num_ch[2]])
    conv31_b = bias_variable([num_ch[2]])
    shared_conv_var = {'conv1_w':conv1_w, 'conv1_b':conv1_b,
                       'conv11_w': conv11_w, 'conv11_b': conv11_b,
                       'conv2_w':conv2_w, 'conv2_b':conv2_b,
                       'conv21_w':conv21_w, 'conv21_b':conv21_b,
                       'conv3_w':conv3_w, 'conv3_b':conv3_b,
                       'conv31_w': conv31_w, 'conv31_b': conv31_b,}
    end_points = voxnet(x,shared_conv_var)


    ## LOSS FUNCTION ##
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=end_points['Logits']))
    # err = tf.reduce_mean(tf.abs(y-end_points['Predictions']))

    ## OPTIMIZER ##
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdadeltaOptimizer(learning_rate)
    grads=optimizer.compute_gradients(cost)
    # for i,(g,v) in enumerate(grads):
    #     if g is not None:
    #         grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
    train_op=optimizer.apply_gradients(grads)


    ## Monitor ##
    overall_acc, overall_acc_update = tf.metrics.accuracy(labels=tf.argmax(y,1), predictions=tf.argmax(end_points['Predictions'],1))
    err = 1. - overall_acc
    # saver = tf.train.Saver() # saves variables learned during training
    logdir, modeldir = creat_dir("voxnet_v3_lr{}".format(lr))
    copyfile('./classifier/voxnet_v3.py', modeldir + '/' + 'voxnet_v3.py')
    saver = tf.train.Saver()
    #saver.restore(sess, "*.ckpt")
    summary_writer = tf.summary.FileWriter(logdir)
    summary_op_train = tf.summary.merge([
        tf.summary.scalar("loss/loss_train", cost),
        tf.summary.scalar("loss/err_last_train", err),
        tf.summary.scalar("lr/lr", learning_rate),
    ])
    summary_op_test = tf.summary.merge([
        tf.summary.scalar("loss/loss_test", cost),
        tf.summary.scalar("loss/err_last_test", err),
    ])



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

    import h5py
    filename = './data/ModelNet/ModelNet10.hdf5'
    f = h5py.File(filename, 'r')
    x_test = np.asarray(f[f.keys()[0]])
    x_train = np.asarray(f[f.keys()[1]])
    y_test = np.asarray(f[f.keys()[2]])
    y_train = np.asarray(f[f.keys()[3]])

    count = 0
    it_per_ep = int( len(x_train) / batch_size )
    for itr in range(5000):
        for i in tqdm(range(it_per_ep)):
            x_input = x_train[i*batch_size : (i + 1)*batch_size]
            y_input = y_train[i*batch_size : (i + 1)*batch_size]
            feed_dict_train = {x: x_input, y: y_input, is_training:True}
            results = sess.run(train_op, feed_dict_train)

            if count % 100 == 0:
                sess.run(tf.local_variables_initializer())
                for ii in range(int( len(x_test) / batch_size) ):
                    x_input = x_test[ii * batch_size:(ii + 1) * batch_size]
                    y_input = y_test[ii * batch_size:(ii + 1) * batch_size]
                    feed_dict_test = {x: x_input, y: y_input, is_training: False}
                    sess.run(overall_acc_update, feed_dict_test)
                summary_test = sess.run(summary_op_test, feed_dict_test)
                summary_writer.add_summary(summary_test, count)
                for ii in range( int(len(x_train) / batch_size) ):
                    x_input = x_train[ii * batch_size:(ii + 1) * batch_size]
                    y_input = y_train[ii * batch_size:(ii + 1) * batch_size]
                    feed_dict_train = {x: x_input, y: y_input, is_training: False}
                    sess.run(overall_acc_update, feed_dict_train)
                summary_train = sess.run(summary_op_train, feed_dict_train)
                summary_writer.add_summary(summary_train, count)
                summary_writer.flush()

            if count % 50000 == 0:
                snapshot_name = "%s_%s" % ('experiment', str(count))
                fn = saver.save(sess, "%s/%s.ckpt" % (modeldir, snapshot_name))
                print("Model saved in file: %s" % fn)
                sess.run(tf.assign(learning_rate, learning_rate * 0.5))
            count += 1
