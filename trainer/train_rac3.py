import tensorflow as tf
from classifier.voxnet import *
from reader.glimpse3 import take_a_3d_glimpse
from utils.utils import creat_dir
from tqdm import tqdm
import numpy as np


def main(cfg):
    vox_size = cfg['vox_size']
    img_dim = cfg['img_dim']
    batch_size = cfg['batch_size']
    num_glimpse = cfg['num_glimpse']
    glimpse_size = cfg['glimpse_size']
    data_path = cfg['data_path']
    lr = cfg['lr']
    n_class = cfg['n_class']

    ## MODEL PARAMETERS ##
    x = tf.placeholder(tf.float32,shape=(batch_size, vox_size, vox_size, vox_size, 1))
    y = tf.placeholder(tf.float32,shape=(batch_size, n_class))

    ## STATE VARIABLES ##
    rr = []
    cs=[0]*num_glimpse # sequence of canvases
    loc=[tf.zeros((batch_size,img_dim))] #starting with a point at the cornor

    ## Build model ##
    num_ch = [16, 32]
    conv1_w = weight_variable([5, 5, 5, 1, num_ch[0]])
    conv1_b = bias_variable([num_ch[0]])
    conv2_w = weight_variable([5, 5, 5, num_ch[0], num_ch[1]])
    conv2_b = bias_variable([num_ch[1]])
    shared_conv_var = {'conv1_w': conv1_w, 'conv1_b': conv1_b, 'conv2_w': conv2_w, 'conv2_b': conv2_b, }

    DO_SHARE=None # workaround for variable_scope(reuse=True)
    sigma = 0.3
    rr += [take_a_3d_glimpse(x[:,:,:,:,0], loc[-1], glimpse_size, delta=1, sigma=sigma)] # first glimpse
    for t in range(num_glimpse):
        cs[t] =  rr[-1] if t==0 else tf.clip_by_value(cs[t-1] + rr[-1],0,1) #canvas
        with tf.variable_scope("VoxNet",reuse=DO_SHARE) as voxnet_scope:
            end_points = voxnet(tf.expand_dims(cs[t],4), shared_conv_var)
        with tf.variable_scope("reader", reuse=DO_SHARE) as reader_scope:
            slim = tf.contrib.slim
            features = end_points['Flatten']
            features = slim.fully_connected(features, 128, activation_fn=tf.sigmoid, scope='fc1')
            loc += [slim.fully_connected(features, 3,activation_fn=tf.sigmoid, scope='fc2')]
            rr += [take_a_3d_glimpse(x[:,:,:,:,0], loc[-1], glimpse_size, delta = 1, sigma = sigma)]
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
    logdir, modeldir = creat_dir("VoxNet_T{}_n{}".format(num_glimpse, glimpse_size))
    saver = tf.train.Saver()
    #saver.restore(sess, "*.ckpt")
    summary_writer = tf.summary.FileWriter(logdir)
    grad1 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Variable:0")[0])[0]))
    grad2 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Variable_2:0")[0])[0]))
    grad3 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="VoxNet/fc3/weights:0")[0])[0]))
    grad4 = tf.reduce_mean(tf.abs(tf.gradients(cost, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="VoxNet/fc4/weights:0")[0])[0]))
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
    import h5py
    filename = data_path+'/ModelNet10.hdf5'
    f = h5py.File(filename, 'r')
    x_test = np.asarray(f[f.keys()[0]])
    x_train = np.asarray(f[f.keys()[1]])
    y_test = np.asarray(f[f.keys()[2]])
    y_train = np.asarray(f[f.keys()[3]])

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
    for epoch in range(5000):
        it_per_ep = len(x_train) / batch_size
        for i in tqdm(range(it_per_ep)):
            x_input = x_train[i*batch_size:(i + 1)*batch_size]
            y_input = y_train[i*batch_size:(i + 1)*batch_size]
            feed_dict_train={x:x_input, y:y_input}
            sess.run(train_op,feed_dict_train)

            if count%50==0:
                train_result = sess.run([cost]+err, feed_dict_train)
                rand_idx = np.random.random_integers(0,len(x_test)-1,size=batch_size)
                x_input = x_test[rand_idx]
                y_input = y_test[rand_idx]
                feed_dict_test = {x: x_input, y: y_input}
                test_result = sess.run([cost]+err, feed_dict_test)
                print("iter=%d : train_cost: %f train_err_last: %f test_cost: %f test_err_last: %f" %
                      (count, train_result[0], train_result[-1], test_result[0], test_result[-1]))
                summary = sess.run(summary_op, feed_dict_train)
                summary_writer.add_summary(summary, count)
                summary_writer.flush()

                # if itr == 0:
                #     feed_dict_train_fix = feed_dict_train
                #     feed_dict_test_fix = feed_dict_test
                # cs_train = sess.run(cs,feed_dict_train_fix)
                # loc_train = sess.run(loc,feed_dict_train_fix)
                # cs_test = sess.run(cs,feed_dict_test_fix)
                # loc_test = sess.run(loc,feed_dict_test_fix)
                # nrow = 30
                # for t in range(num_glimpse):
                #     canvas_train = cs_train[t][:nrow] if t == 0 else np.concatenate([canvas_train, cs_train[t][:nrow]], 0)
                #     canvas_test = cs_test[t][:nrow] if t == 0 else np.concatenate([canvas_test, cs_test[t][:nrow]], 0)
                # all_img_out = 255 * np.concatenate([feed_dict_train_fix[x][:nrow], canvas_train, feed_dict_test_fix[x][:nrow], canvas_test])
                # im = save_image(np.expand_dims(all_img_out,3), '{}/itr{}.png'.format(logdir, itr), nrow=nrow)
                #
                # img_idx = 0
                # arr = np.asarray([cs_test[idx][img_idx] for idx in range(len(cs_test))])
                # center = np.asarray([loc_test[idx][img_idx] for idx in range(len(loc_test))])
                # cx_i = (center[:-1, 0] + 1) / 2 * 28
                # cy_i = (center[:-1, 1] + 1) / 2 * 28
                # prediction = sess.run(y_hat, feed_dict_test_fix)[img_idx].transpose()
                # filename =  '{}/itr{}_pred{}.png'.format(logdir, itr,feed_dict_test_fix[y][img_idx])
                # filename =  '{}/itr{}_pred{}.png'.format(logdir, itr,feed_dict_test_fix[y][img_idx])
                # ram_mnist_viz(arr, cy_i - np.ceil(glimpse_size / 2.), cx_i - np.ceil(glimpse_size / 2.), prediction, filename)

            if count%5000==1:
                sess.run( tf.assign(learning_rate, learning_rate * 0.5) )

            if count%10000==1:
                snapshot_name = "%s_%s" % ('experiment', str(count))
                fn = saver.save(sess, "%s/%s.ckpt" % (modeldir, snapshot_name))
                print("Model saved in file: %s" % fn)
            count += 1

if __name__ == "__main__":


    cfg = {'batch_size': 128,
           'img_dim': 3,
           'vox_size': 32,
           'n_class': 10,
           'num_glimpse': 8,
           'glimpse_size': 3,
           'data_path': '../data/ModelNet',
           'lr': 1e-3,
           }
    main(cfg)