import os.path
import urllib.request
import struct
import numpy as np
import gzip

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

# http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

def maybe_download(filename, work_directory):
      """Download the data from Yann's website, unless it's already here."""
      filename = filename + ".gz"
      if not os.path.exists(work_directory):
        os.mkdir(work_directory)
      filepath = os.path.join(work_directory, filename)
      if not os.path.exists(filepath):
        print("Downloading", filename)
        url = SOURCE_URL + filename
        filepath, _ = urllib.request.urlretrieve(url, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
      return filepath


def load_mnist(dataset="training", digits=np.arange(10), path="./mnist_files", size=60000):
    if dataset == "training":
        fname = 'train-images-idx3-ubyte'
        fname_img = maybe_download(fname, path)
        fname = 'train-labels-idx1-ubyte'
        fname_lbl = maybe_download(fname, path)
    elif dataset == "testing":
        fname = 't10k-images-idx3-ubyte'
        fname_img = maybe_download(fname, path)
        fname = 't10k-labels-idx1-ubyte'
        fname_lbl = maybe_download(fname, path)
    else:
        raise ValueError("Dataset must be 'testing' or 'training'")

    with gzip.open(fname_lbl, 'rb') as flbl:
        magic, size = struct.unpack(">II", flbl.read(8))
        lbl = np.frombuffer(flbl.read(), dtype=np.ubyte)

    with gzip.open(fname_img, 'rb') as fimg:
        magic, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.frombuffer(fimg.read(), dtype=np.ubyte).reshape(size, rows, cols)

    return img, lbl
