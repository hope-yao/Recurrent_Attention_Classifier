# import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
from utils import *
from lenet_slim import lenet
from glimpse_2d import take_a_2d_glimpse
from glimpse_3d import take_a_3d_glimpse



def main(cfg):
    img_size = cfg['img_size']
    img_dim = cfg['img_dim']
    batch_size = cfg['batch_size']
    num_glimpse = cfg['num_glimpse']
    glimpse_size = cfg['glimpse_size']
    data_path = cfg['data_path']
    lr = cfg['lr']
    n_class = cfg['n_class']

    ## MODEL PARAMETERS ##
    x = tf.placeholder(tf.float32,shape=(batch_size, img_size, img_size))
    y = tf.placeholder(tf.float32,shape=(batch_size, n_class))

    ## STATE VARIABLES ##
    rr = []
    cs=[0]*num_glimpse # sequence of canvases
    loc=[tf.ones((batch_size,img_dim))] #starting with a point at the cornor

    ## Build model ##
    DO_SHARE=None # workaround for variable_scope(reuse=True)
    rr += [take_a_2d_glimpse(x, loc[-1], glimpse_size, delta=1, sigma=0.4)] # first glimpse
    for t in range(num_glimpse):
        cs[t] =  rr[-1] if t==0 else tf.clip_by_value(cs[t-1] + rr[-1],0,1) #canvas
        with tf.variable_scope("LeNet",reuse=DO_SHARE) as lenet_scope:
            mid_output, end_points = lenet(tf.expand_dims(cs[t],3), num_classes=10, is_training=True,
                                  dropout_keep_prob=0.99,
                                  scope='LeNet')
        with tf.variable_scope("reader", reuse=DO_SHARE) as reader_scope:
            slim = tf.contrib.slim
            features = end_points['Flatten']
            features = slim.fully_connected(features, 64, activation_fn=None, scope='fc1')
            loc += [slim.fully_connected(features, 2,activation_fn=None, scope='fc2')]
            rr += [take_a_2d_glimpse(x, loc[-1], glimpse_size, delta = 1, sigma = 0.4)]
        log_y_hat_i = tf.expand_dims(end_points['Logits'],2)
        log_y_hat = log_y_hat_i if t==0 else tf.concat([log_y_hat,log_y_hat_i],2)
        y_hat_i = tf.expand_dims(end_points['Predictions'],2)
        y_hat = y_hat_i if t==0 else tf.concat([y_hat,y_hat_i],2)
        DO_SHARE = True# workaround for variable_scope(reuse=True)

    ## LOSS FUNCTION ##
    loss_classify = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=log_y_hat[:,:,-1]))
    cost = loss_classify #+ loss_loc
    err = [tf.reduce_mean(tf.abs(y-y_hat[:,:,i])) for i in range(num_glimpse)]

    ## OPTIMIZER ##
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    grads=optimizer.compute_gradients(cost)
    # for i,(g,v) in enumerate(grads):
    #     if g is not None:
    #         grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
    train_op=optimizer.apply_gradients(grads)


    ## Monitor ##
    # saver = tf.train.Saver() # saves variables learned during training
    logdir, modeldir = creat_dir("drawtf_T{}_n{}".format(num_glimpse, glimpse_size))
    saver = tf.train.Saver()
    #saver.restore(sess, "*.ckpt")
    summary_writer = tf.summary.FileWriter(logdir)
    grad1 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LeNet/conv1/weights:0")[0])[0]))
    grad2 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LeNet/conv2/weights:0")[0])[0]))
    grad3 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LeNet/fc3/weights:0")[0])[0]))
    grad4 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LeNet/fc4/weights:0")[0])[0]))
    grad5 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="reader/fc1/weights:0")[0])[0]))
    grad6 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="reader/fc2/weights:0")[0])[0]))
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
        x_train = np.reshape(x_train,(x_train.shape[0],img_size,img_size))
        feed_dict_train={x:x_train, y:y_train}
        results = sess.run(train_op,feed_dict_train)

        if itr%100==0:
            x_test, y_test = test_data.next_batch(batch_size)  # xtrain is (batch_size x img_size)
            x_test = np.reshape(x_test, (x_test.shape[0], img_size, img_size))
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
            for t in range(num_glimpse):
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
            ram_mnist_viz(arr, cy_i - np.ceil(glimpse_size / 2.), cx_i - np.ceil(glimpse_size / 2.), prediction, filename)

        if itr%10000==0:
            sess.run( tf.assign(learning_rate, learning_rate * 0.5) )

        if itr%10000==0:
            snapshot_name = "%s_%s" % ('experiment', str(itr))
            fn = saver.save(sess, "%s/%s.ckpt" % (modeldir, snapshot_name))
            print("Model saved in file: %s" % fn)

if __name__ == "__main__":


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