#!/usr/bin/env python

""""
Classifier version of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

"""
import tensorflow as tf
eps=1e-8 # epsilon for numerical stability


## get filter bank matrices##
def filter_banks(loc, read_n, img_size, delta_, sigma_):
    batch_size = loc.shape[0].value
    delta = delta_*tf.ones((batch_size,1), 'float32')
    sigma2 = sigma_*tf.ones((batch_size,1), 'float32')
    gx_,gy_=tf.split(loc,2,1)
    gx=(img_size+1)/2*(gx_+1)
    gy=(img_size+1)/2*(gy_+1)

    grid_i = tf.reshape(tf.cast(tf.range(read_n), tf.float32), [1, -1])
    mu_x = gx + (grid_i - read_n / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - read_n / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(img_size), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(img_size), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, read_n, 1])
    mu_y = tf.reshape(mu_y, [-1, read_n, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.tile(tf.expand_dims(tf.reduce_mean(eps+tf.reduce_sum(Fx,2,keep_dims=True),1),1),(1,read_n,1))
    Fy=Fy/tf.tile(tf.expand_dims(tf.reduce_mean(eps+tf.reduce_sum(Fy,2,keep_dims=True),1),1),(1,read_n,1))
    return Fx,Fy


## READ ##
def read_attn_const_gamma(x, Fx, Fy):
    Fxt = tf.transpose(Fx, perm=[0, 2, 1])
    x = tf.matmul(Fy, tf.matmul(x, Fxt))
    return x

## WRITE ##
def write_attn_const_gamma(glimpse, Fx, Fy):
    Fyt=tf.transpose(Fy,perm=[0,2,1])
    wr=tf.matmul(Fyt,tf.matmul(glimpse,Fx))
    return wr


def take_a_2d_glimpse(x, loc, read_n, delta = 1.0, sigma = 1.0):
    img_size = x.shape[1].value
    Fx,Fy = filter_banks( loc, read_n, img_size, delta_=delta, sigma_=sigma)
    x = read_attn_const_gamma(x, Fx, Fy)
    x = write_attn_const_gamma(x, Fx, Fy)
    return x
