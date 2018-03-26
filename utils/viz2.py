import math
import numpy as np
from PIL import Image


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

def mnist_viz(arr,cx_i,cy_i,y,filename):
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
    # fig = plt.figure(figsize=(6*num_glimpse,4.5*num_glimpse))
    # gs = gridspec.GridSpec(2,num_glimpse,height_ratios=[1]*num_glimpse+[2]*num_glimpse,
    #                        width_ratios=[1]*num_glimpse+[3]*num_glimpse)
    fig = plt.figure(figsize=(6*1,4.5*1))
    gs = gridspec.GridSpec(2,num_glimpse)

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
