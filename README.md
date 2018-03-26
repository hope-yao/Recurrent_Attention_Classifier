# draw
![acc](../master/assests/rac_acc.png)

tensorflow-gpu == 1.0.1
## Usage

`python draw.py --data_dir=/tmp/draw` downloads the binarized MNIST dataset to /tmp/draw/mnist and trains the DRAW model with attention enabled for both reading and writing. After training, output data is written to `/tmp/draw/draw_data.npy`

You can visualize the results by running the script `python plot_data.py <prefix> <output_data>`

For example, 

`python myattn /tmp/draw/draw_data.npy`

To run training without attention, do:

`python draw.py --working_dir=/tmp/draw --read_attn=False --write_attn=False`

To download the preprocessed ModelNet10 dataset used in this code, do:
```
$ wget
```

## Network structure
![network](../master/assests/attention_network.png)


## Results
Note that sigma has a big influence on the convergence. There will be gradient vanishing issue for the reader if sigma is too small.

## ToDO
