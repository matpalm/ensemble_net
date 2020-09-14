import tensorflow as tf
import tensorflow_datasets as tfds
from functools import lru_cache
import numpy as np
import jax.numpy as jnp
from tensorflow.data.experimental import AUTOTUNE


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
               .batch(batch_size)
               .shuffle(1024))
    return tfds.as_numpy(dataset)


def validation_dataset(batch_size):
    # 2700 records
    # [293, 307, 335, 258, 253, 194, 239, 284, 243, 294]
    return _non_training_dataset(batch_size, 'train[80%:90%]')


def test_dataset(batch_size):
    # 2700 records
    # [307, 300, 296, 221, 262, 216, 251, 296, 250, 301]
    return _non_training_dataset(batch_size, 'train[90%:]')


def training_dataset(batch_size, num_inputs=1):
    dataset = (tfds.load('eurosat/rgb', split='train[:80%]',
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


if __name__ == '__main__':
    from PIL import Image

    def pil_img_from_array(array):
        return Image.fromarray((array * 255.0).astype(np.uint8))

    SAMPLE_GRID_SIZE = 5   # each sample will have SGSxSGS images
    NUM_CLASSES = 10

    class Collage(object):
        def __init__(self):
            self.collage = Image.new('RGB', (64*SAMPLE_GRID_SIZE,
                                             64*SAMPLE_GRID_SIZE))
            self.insert_idx = 0
            self.full = False

        def add(self, np_array):
            # return true on last pasted
            if self.full:
                return False
            x = int((self.insert_idx // SAMPLE_GRID_SIZE) * 64)
            y = int((self.insert_idx % SAMPLE_GRID_SIZE) * 64)
            self.collage.paste(pil_img_from_array(np_array), (x, y))
            self.insert_idx += 1
            if self.insert_idx == SAMPLE_GRID_SIZE * SAMPLE_GRID_SIZE:
                self.full = True
            return self.full

    collages = [Collage() for _ in range(NUM_CLASSES)]

    num_collages_full = 0
    for imgs, labels in training_dataset(batch_size=128):
        for img, label in zip(imgs, labels):
            if collages[label].add(img):
                num_collages_full += 1
                if num_collages_full == 10:
                    break

    class_names = ['Annual_Crop', 'Forest', 'Herbaceous_Vegetation', 'Highway',
                   'Industrial_Buildings', 'Pasture', 'Permanent_Crop',
                   'Residential_Buildings', 'River', 'Sea_Lake']
    for i, c in enumerate(collages):
        c.collage.save("collage.%02d.%s.png" % (i, class_names[i]))
