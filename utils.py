from __future__ import print_function

import os
import math
import json
import logging
from PIL import Image
from datetime import datetime
import numpy as np


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rank(array):
    return len(array.shape)

def make_grid(tensor, nrow=8, padding=4,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.ones([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8) *127
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=4,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)
    return im

import tensorflow as tf
# def list2tensor(alist,dim=0):
#     for i, list_i in enumerate(alist):
#         if i == 0:
#             atensor = list_i
#         else:
#             atensor = tf.concat([atensor, list_i], dim)
#     return atensor


import dateutil.tz

def creat_dir(network_type):
    """code from on InfoGAN"""
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_log_dir = "logs/" + network_type
    exp_name = network_type + "_%s" % timestamp
    log_dir = os.path.join(root_log_dir, exp_name)

    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_model_dir = "models/" + network_type
    exp_name = network_type + "_%s" % timestamp
    model_dir = os.path.join(root_model_dir, exp_name)

    for path in [log_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    return log_dir, model_dir




def ram_mnist_viz(arr,cx_i,cy_i,y,filename):
    '''
    :param arr: canvas, (t,length,width)
    :param cx_i: (t,1)
    :param cy_i: (t,1)
    :param y: probability of the classification (t,n_class)
    :return: figure of a single digit
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.patches import Rectangle
    mnist = [0,1,2,3,4,5,6,7,8,9]
    num_glimpse = arr.shape[0]
    fig = plt.figure(figsize=(6*num_glimpse,4.5*num_glimpse))
    gs = gridspec.GridSpec(2,num_glimpse,height_ratios=[1]*num_glimpse+[2]*num_glimpse,
                           width_ratios=[1]*num_glimpse+[3]*num_glimpse)

    for i in range(num_glimpse):
        ax0=plt.subplot(gs[i])
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.axis('off')

        for ii,jj in zip(y[i], mnist):
            ax0.annotate(str(jj),xy=(jj+0.1,ii+0.03))

        ax0.bar(mnist,y[i],width=1,color='g',linewidth=0)
        ax0.set_ylim(0,1.1)

        ax0=plt.subplot(gs[num_glimpse+i])
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.axis('off')
        #ax0 = plt.gca()
        ax0.add_patch(Rectangle((cy_i[i], cx_i[i]), 4, 4, edgecolor="red", linewidth=2,fill=False))
        if i > 0:
            for ii in range(i,0,-1):
                ax0.add_patch(Rectangle((cy_i[ii-1],cx_i[ii-1]), 4, 4, edgecolor="blue", linewidth=1,fill=False))

        ax0.imshow(arr[i].reshape(28,28),'gray')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(filename)  # save the figure to file
    plt.close(fig)  # close the figure
