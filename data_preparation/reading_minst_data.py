""" A function that can read MNIST's idx file format into numpy arrays.
    The MNIST data files can be downloaded from here:

    http://yann.lecun.com/exdb/mnist/
    This relies on the fact that the MNIST dataset consistently uses
    unsigned char types with their data segments.
"""

import struct

import numpy as np


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def img_show(img):
    from PIL import Image
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

if __name__ == '__main__':
    file_name = r"C:\Users\maaya\Documents\minst_data\train-images.idx3-ubyte"
    data = read_idx(file_name)
    img_show(data[20])
