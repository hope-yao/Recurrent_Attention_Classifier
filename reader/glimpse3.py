#!/usr/bin/env python

""""
Classifier version of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

"""
import tensorflow as tf
eps=1e-8 # epsilon for numerical stability

## BUILD MODEL ##
def attn_window_const_gamma(loc, read_n, img_size, delta_, sigma_):
    batch_size = loc.shape[0].value
    delta = delta_*tf.ones((batch_size,1), 'float32')
    sigma2 = sigma_*tf.ones((batch_size,1), 'float32')
    gx_, gy_, gz_ = tf.split(loc,3,1)
    gx=(img_size+1)/2*(gx_+1)
    gy=(img_size+1)/2*(gy_+1)
    gz=(img_size+1)/2*(gz_+1)

    glimpse_grid = tf.reshape(tf.cast(tf.range(read_n), tf.float32), [1, -1])
    mu_x = gx + (glimpse_grid - read_n / 2 - 0.5) * delta # eq 19
    mu_y = gy + (glimpse_grid - read_n / 2 - 0.5) * delta # eq 20
    mu_z = gz + (glimpse_grid - read_n / 2 - 0.5) * delta # eq 20
    vox_gird = tf.reshape(tf.cast(tf.range(img_size), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, read_n, 1])
    mu_y = tf.reshape(mu_y, [-1, read_n, 1])
    mu_z = tf.reshape(mu_z, [-1, read_n, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((vox_gird - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((vox_gird - mu_y) / (2*sigma2))) # batch x N x B
    Fz = tf.exp(-tf.square((vox_gird - mu_z) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.tile(tf.expand_dims(tf.reduce_mean(eps+tf.reduce_sum(Fx,2,keep_dims=True),1),1),(1,read_n,1))
    Fy=Fy/tf.tile(tf.expand_dims(tf.reduce_mean(eps+tf.reduce_sum(Fy,2,keep_dims=True),1),1),(1,read_n,1))
    Fz=Fz/tf.tile(tf.expand_dims(tf.reduce_mean(eps+tf.reduce_sum(Fz,2,keep_dims=True),1),1),(1,read_n,1))
    return Fx,Fy, Fz


## READ ##
def read_attn_const_gamma(x, Fx, Fy, Fz):
    _, read_n, _ = Fx.get_shape().as_list()
    Fxt = tf.transpose(Fx, perm=[0, 2, 1])
    bs, height, width, depth = x.get_shape().as_list()

    i1 = tf.reshape(x, (bs, height, width * depth))
    iy = tf.reshape(tf.matmul(Fy,i1),(bs,read_n,width,depth))
    i2 = tf.reshape(tf.transpose(iy,(0,1,3,2)),(bs,read_n*depth,width))
    ix = tf.transpose(tf.reshape(tf.matmul(i2,Fxt),(bs,read_n,depth,read_n)),(0,1,3,2))
    i3 = tf.reshape(tf.transpose(ix,(0,3,1,2)),(bs,depth,read_n*read_n))
    iz = tf.matmul(Fz,i3)
    ii = tf.reshape(iz,(bs,read_n,read_n,read_n))

    return ii

## WRITE ##
def write_attn_const_gamma(glimpse, Fx, Fy, Fz):
    bs, read_n, vox_size = Fx.get_shape().as_list()
    Fzt = tf.transpose(Fz, perm=[0, 2, 1])

    iz = tf.matmul(Fzt, tf.reshape(tf.transpose(glimpse,(0,3,1,2)),(bs,read_n,read_n*read_n)))
    i3 = tf.reshape(iz,(bs,vox_size,read_n,read_n))#order:z,x,y; size:n,r,r
    ix = tf.matmul(tf.reshape(tf.transpose(i3,(0,1,3,2)),(bs,vox_size*read_n,read_n)), Fx)
    i2 = tf.reshape(ix,(bs,vox_size,read_n,vox_size))#order:z,y,x; size:n,r,n
    iy = tf.matmul(tf.reshape(tf.transpose(i2,(0,1,3,2)),(bs, vox_size*vox_size,read_n)), Fy)
    i1 = tf.reshape(iy,(bs,vox_size,vox_size,vox_size)) #order:z,x,y; size:n,n,n
    ii = tf.transpose(i1,(0,2,3,1)) #order:x,y,z
    return ii


def take_a_3d_glimpse(x, loc, read_n, delta = 1.0, sigma = 1.0):
    img_size = x.shape[1].value
    Fx, Fy, Fz = attn_window_const_gamma( loc, read_n, img_size, delta_=delta, sigma_=sigma)
    glimpse = read_attn_const_gamma(x, Fx, Fy, Fz)
    canvase = write_attn_const_gamma(glimpse, Fx, Fy, Fz)
    return canvase


if __name__ == '__main__':
    x = tf.ones((128,32,32,32))
    loc = 0.5*tf.ones((128,3))
    read_n = 3
    glimpse = take_a_3d_glimpse(x,loc,read_n,sigma=0.4)


    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    from utils.viz3 import viz3
    viz3(glimpse[0].eval(session=sess),1)
    print('testing done...')