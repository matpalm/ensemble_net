import tensorflow as tf
import tensorflow_datasets as tfds
from functools import lru_cache
import numpy as np


def _x_to_float(x, y):
    x = tf.cast(x, tf.float32) / 255
    return x, y


@lru_cache()
def entire_split(ds_split):
    x, y = tfds.load('eurosat/rgb', split=ds_split, shuffle_files=False,
                     batch_size=-1, as_supervised=True)
    x, y = _x_to_float(x, y)
    return np.array(x), np.array(y)


def dataset(split, batch_size=16):
    if split == 'train':  # 21600 records
        ds_split = 'train[:80%]'
    elif split == 'validate':  # 2700 records
        ds_split = 'train[80%:90%]'
    elif split == 'test':  # 2700 records
        ds_split = 'train[90%:]'
    else:
        raise Exception("unexpected split", split)

    if split == 'validate' or split == 'test':
        # return entire dataset in one (cached) pair of numpy arrays.
        return entire_split(ds_split)
    else:
        # returned batched (shuffled) iterator.
        dataset = (tfds.load('eurosat/rgb', split=ds_split,
                             shuffle_files=True, batch_size=batch_size,
                             as_supervised=True)
                   .map(_x_to_float)
                   .prefetch(tf.data.experimental.AUTOTUNE))
        return tfds.as_numpy(dataset)
