#!/usr/bin/env python

""""
Classifier version of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os

from lenet_slim import lenet


tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn",True, "enable attention for writer")
FLAGS = tf.flags.FLAGS

## MODEL PARAMETERS ## 
eps=1e-8 # epsilon for numerical stability


batch_size=128 # training minibatch size
A,B = 28,28 # image width,height
img_size = B*A # the canvas size
n_class = 10 # ten class of digits
x = tf.placeholder(tf.float32,shape=(batch_size, A, B)) # input (batch_size * img_size)
y = tf.placeholder(tf.float32,shape=(batch_size, n_class)) # input (batch_size * img_size)


## BUILD MODEL ##

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def filterbank(gx, gy, sigma2,delta, read_n):
    grid_i = tf.reshape(tf.cast(tf.range(read_n), tf.float32), [1, -1])
    mu_x = gx + (grid_i - read_n / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - read_n / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, read_n, 1])
    mu_y = tf.reshape(mu_y, [-1, read_n, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.tile(tf.expand_dims(tf.reduce_mean(eps+tf.reduce_sum(Fx,2,keep_dims=True),1),1),(1,read_n,1))
    Fy=Fy/tf.tile(tf.expand_dims(tf.reduce_mean(eps+tf.reduce_sum(Fy,2,keep_dims=True),1),1),(1,read_n,1))
    return Fx,Fy

def attn_window_const_gamma(scope,loc, read_n, delta_, sigmas_ ):
    delta = delta_*tf.ones((batch_size,1), 'float32')
    sigmas = sigmas_*tf.ones((batch_size,1), 'float32')
    gx_,gy_=tf.split(loc,2,1)
    gx=(A+1)/2*(gx_+1)
    gy=(B+1)/2*(gy_+1)
    return filterbank(gx,gy,sigmas,delta,read_n)

## READ ##
def read_attn_const_gamma(x,loc, read_n, delta_=0.1, sigma_=0.1):
    Fx,Fy = attn_window_const_gamma("read",loc,read_n, delta_, sigma_)
    def filter_img(img,Fx,Fy, N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,B,A])
        glimpse=tf.matmul(Fy,tf.matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse
    x = filter_img(x,Fx,Fy,read_n) # batch x (read_n*read_n)
    x = tf.reshape(x,(batch_size, read_n, read_n))
    return x

read = read_attn_const_gamma

## WRITER ##
def write_attn_const_gamma(glimpse, loc, read_n, delta_=0.1, sigma_=0.1):
    Fx,Fy = attn_window_const_gamma("write", loc, read_n, delta_, sigma_)
    Fyt=tf.transpose(Fy,perm=[0,2,1])
    wr=tf.matmul(Fyt,tf.matmul(glimpse,Fx))
    wr=tf.reshape(wr,[batch_size,B,A])
    return wr

write = write_attn_const_gamma

def take_a_glimpse(img, loc, read_n, delta = 1.0, sigma = 1.0):
    x = read_attn_const_gamma(img,loc, read_n, delta_=delta,sigma_=sigma)
    x = write_attn_const_gamma(x,loc, read_n,delta_=delta,sigma_=sigma)
    return x


## STATE VARIABLES ##
rr = []
T = 5 # MNIST generation sequence length
cs=[0]*T # sequence of canvases
loc=[tf.ones((batch_size,2))] #(-1,1)

## Build model ##
read_n = 3 # read glimpse grid width/height
write_n = read_n# write glimpse grid width/height
DO_SHARE=None # workaround for variable_scope(reuse=True)
for t in range(T):
    rr += [take_a_glimpse(x, loc[-1], read_n, delta = 1, sigma = 0.4)]
    cs[t] =  rr[-1] if t==0 else tf.clip_by_value(cs[t-1] + rr[-1],0,1) #canvas
    with tf.variable_scope("LeNet",reuse=DO_SHARE) as lenet_scope:
        mid_output, end_points = lenet(tf.expand_dims(cs[t],3), num_classes=10, is_training=True,
                              dropout_keep_prob=0.99,
                              scope='LeNet')
        features = end_points['Flatten']
        slim = tf.contrib.slim
        # features = slim.fully_connected(features, 1024, activation_fn=None)
    with tf.variable_scope("LOC", reuse=DO_SHARE) as Loc_scope:
        features = slim.fully_connected(features, 64, activation_fn=None, scope='fc1')
        loc += [slim.fully_connected(features, 2,activation_fn=None, scope='fc2')]
    log_y_hat_i = tf.expand_dims(end_points['Logits'],2)
    log_y_hat = log_y_hat_i if t==0 else tf.concat([log_y_hat,log_y_hat_i],2)
    y_hat_i = tf.expand_dims(end_points['Predictions'],2)
    y_hat = y_hat_i if t==0 else tf.concat([y_hat,y_hat_i],2)
    DO_SHARE = True# workaround for variable_scope(reuse=True)

## LOSS FUNCTION ##
loss_classify = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=log_y_hat[:,:,-1]))
# loss_loc = tf.reduce_mean(tf.abs(tf.clip_by_value(loc,1,-1)))
cost = loss_classify #+ loss_loc
err = [tf.reduce_mean(tf.abs(y-y_hat[:,:,i])) for i in range(T)]
grad1 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LeNet/conv1/weights:0")[0])[0]))
grad2 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LeNet/conv2/weights:0")[0])[0]))
grad3 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LeNet/fc3/weights:0")[0])[0]))
grad4 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LeNet/fc4/weights:0")[0])[0]))
grad5 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LOC/fc1/weights:0")[0])[0]))
grad6 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LOC/fc2/weights:0")[0])[0]))

## OPTIMIZER ##
learning_rate = tf.Variable(1e-3) # learning rate for optimizer
optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads=optimizer.compute_gradients(cost)
# for i,(g,v) in enumerate(grads):
#     if g is not None:
#         grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op=optimizer.apply_gradients(grads)

## RUN TRAINING #c#
data_directory = os.path.join(FLAGS.data_dir, "mnist")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data
test_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).test # binarized (0-1) mnist data

# saver = tf.train.Saver() # saves variables learned during training
from utils import *
logdir, modeldir = creat_dir("drawtf_T{}_n{}".format(T, read_n))

FLAGS = tf.app.flags.FLAGS
tfconfig = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
init = tf.global_variables_initializer()
sess.run(init)
#saver.restore(sess, "/tmp/draw/drawmodel.ckpt") # to restore from model, uncomment this line

summary_writer = tf.summary.FileWriter(logdir)
summary_op = tf.summary.merge([
    tf.summary.scalar("loss/loss", cost),

    tf.summary.scalar("err/err_last", err[-1]),

    tf.summary.scalar("lr/lr", learning_rate),

    tf.summary.scalar("grad/grad1", grad1),
    tf.summary.scalar("grad/grad2", grad2),
    tf.summary.scalar("grad/grad3", grad3),
    tf.summary.scalar("grad/grad4", grad4),
    tf.summary.scalar("grad/grad5", grad5),
    tf.summary.scalar("grad/grad6", grad6),
])

train_iters=500000
for itr in range(train_iters):
    x_train,y_train = train_data.next_batch(batch_size) # xtrain is (batch_size x img_size)
    x_train = np.reshape(x_train,(x_train.shape[0],A,B))
    feed_dict_train={x:x_train, y:y_train}
    results = sess.run(train_op,feed_dict_train)

    if itr%100==0:
        x_test, y_test = test_data.next_batch(batch_size)  # xtrain is (batch_size x img_size)
        x_test = np.reshape(x_test, (x_test.shape[0], A, B))
        feed_dict_test = {x: x_test, y: y_test}
        train_result = sess.run([cost]+err, feed_dict_train)
        test_result = sess.run([cost]+err, feed_dict_test)
        print("iter=%d : train_cost: %f train_err_last: %f test_cost: %f test_err_last: %f" %
              (itr, train_result[0], train_result[-1], test_result[0], test_result[-1]))
        summary = sess.run(summary_op, feed_dict_train)
        summary_writer.add_summary(summary, itr)
        summary_writer.flush()

        if itr == 0:
            feed_dict_train_fix = feed_dict_train
            feed_dict_test_fix = feed_dict_test
        cs_train = sess.run(cs,feed_dict_train_fix)
        loc_train = sess.run(loc,feed_dict_train_fix)
        cs_test = sess.run(cs,feed_dict_test_fix)
        loc_test = sess.run(loc,feed_dict_test_fix)
        nrow = 30
        for t in range(T):
            canvas_train = cs_train[t][:nrow] if t == 0 else np.concatenate([canvas_train, cs_train[t][:nrow]], 0)
            canvas_test = cs_test[t][:nrow] if t == 0 else np.concatenate([canvas_test, cs_test[t][:nrow]], 0)
        all_img_out = 255 * np.concatenate([feed_dict_train_fix[x][:nrow], canvas_train, feed_dict_test_fix[x][:nrow], canvas_test])
        im = save_image(np.expand_dims(all_img_out,3), '{}/itr{}.png'.format(logdir, itr), nrow=nrow)

        img_idx = 0
        arr = np.asarray([cs_test[idx][img_idx] for idx in range(len(cs_test))])
        center = np.asarray([loc_test[idx][img_idx] for idx in range(len(loc_test))])
        cx_i = (center[:-1, 0] + 1) / 2 * 28
        cy_i = (center[:-1, 1] + 1) / 2 * 28
        prediction = sess.run(y_hat, feed_dict_test_fix)[img_idx].transpose()
        filename =  '{}/itr{}_pred{}.png'.format(logdir, itr,feed_dict_test_fix[y][img_idx])
        filename =  '{}/itr{}_pred{}.png'.format(logdir, itr,feed_dict_test_fix[y][img_idx])
        ram_mnist_viz(arr, cy_i - np.ceil(read_n / 2.), cx_i - np.ceil(read_n / 2.), prediction, filename)

    if itr%10000==0:
        sess.run( tf.assign(learning_rate, learning_rate * 0.5) )