import tensorflow as tf
import tensorflow_datasets as tfds
from functools import lru_cache
import numpy as np
import jax.numpy as jnp
from tensorflow.data.experimental import AUTOTUNE
import logging


def _convert_dtype(x):
    return tf.cast(x, tf.float32) / 255


@tf.autograph.experimental.do_not_convert
def _augment_and_convert_dtype(x, y):
    # rotate 0, 90, 180 or 270 deg
    k = tf.random.uniform([], 0, 3, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    # flip L/R 50% time
    x = tf.image.random_flip_left_right(x)
    # convert to float
    x = _convert_dtype(x)
    # colour distortion
    x = tf.image.random_saturation(x, 0.5, 1.5)
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y


def _non_training_dataset(batch_size, ds_split):

    @tf.autograph.experimental.do_not_convert
    def _convert_image_dtype(x, y):
        return _convert_dtype(x), y

    dataset = (tfds.load('eurosat/rgb', split=ds_split,
                         as_supervised=True)
               .map(_convert_image_dtype, num_parallel_calls=AUTOTUNE)
               .batch(batch_size))
    return tfds.as_numpy(dataset)


def validation_dataset(batch_size):
    # 2700 records
    # [293, 307, 335, 258, 253, 194, 239, 284, 243, 294]
    return _non_training_dataset(batch_size, 'train[80%:90%]')


def test_dataset(batch_size):
    # 2700 records
    # [307, 300, 296, 221, 262, 216, 251, 296, 250, 301]
    return _non_training_dataset(batch_size, 'train[90%:]')


def training_dataset(batch_size, num_inputs=1, sample_data=False):

    if sample_data:
        logging.warn("using small sample_data for training")
        split = 'train[:5%]'
    else:
        split = 'train[:80%]'

    dataset = (tfds.load('eurosat/rgb', split=split,
                         as_supervised=True)
               .map(_augment_and_convert_dtype, num_parallel_calls=AUTOTUNE)
               .shuffle(1024))

    if num_inputs == 1:
        dataset = dataset.batch(batch_size)
    else:
        @tf.autograph.experimental.do_not_convert
        def _reshape_inputs(x, y):
            _b, h, w, c = x.shape
            x = tf.reshape(x, (num_inputs, batch_size, h, w, c))
            y = tf.reshape(y, (num_inputs, batch_size))
            return x, y
        dataset = dataset.batch(batch_size * num_inputs, drop_remainder=True)
        dataset = dataset.map(_reshape_inputs)
        pass

    dataset = dataset.prefetch(AUTOTUNE)
    return tfds.as_numpy(dataset)
