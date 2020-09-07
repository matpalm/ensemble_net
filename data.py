import tensorflow as tf
import tensorflow_datasets as tfds
from functools import lru_cache
import numpy as np


def _convert_dtype(x):
    return tf.cast(x, tf.float32) / 255


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


@lru_cache()
def entire_split(ds_split):
    x, y = tfds.load('eurosat/rgb', split=ds_split, shuffle_files=False,
                     batch_size=-1, as_supervised=True)
    return np.array(_convert_dtype(x)), np.array(y)


def dataset(split, batch_size=16):
    if split == 'train':  # 21600 records
        ds_split = 'train[:80%]'
    elif split == 'validate':
        # 2700 records
        # [293, 307, 335, 258, 253, 194, 239, 284, 243, 294]
        ds_split = 'train[80%:90%]'
    elif split == 'test':
        # 2700 records
        # [307, 300, 296, 221, 262, 216, 251, 296, 250, 301]
        ds_split = 'train[90%:]'
    else:
        raise Exception("unexpected split", split)

    if split == 'validate' or split == 'test':
        # return entire dataset in one (cached) pair of numpy arrays.
        return entire_split(ds_split)
    else:
        # returned batched (shuffled) iterator.
        dataset = (tfds.load('eurosat/rgb', split=ds_split, as_supervised=True)
                   .map(_augment_and_convert_dtype)
                   .shuffle(1024)
                   .batch(batch_size)
                   .prefetch(tf.data.experimental.AUTOTUNE))
        return tfds.as_numpy(dataset)


if __name__ == '__main__':
    from PIL import Image

    def pil_img_from_array(array):
        return Image.fromarray((array * 255.0).astype(np.uint8))

    B = 4
    for imgs, labels in dataset('train', batch_size=B*B):
        break
    collage = Image.new('RGB', (64*B, 64*B))
    for i in range(B*B):
        r, c = i//B, i % B
        collage.paste(pil_img_from_array(imgs[i]), (r*64, c*64))
    collage.resize((64*B*3, 64*B*3)).show()
