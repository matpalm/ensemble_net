import tensorflow as tf
import tensorflow_datasets as tfds


def dataset(split, batch_size):
    if split == 'train':  # 21600 records
        ds_split = 'train[:80%]'
    elif split == 'validate':  # 2700 records
        ds_split = 'train[80%:90%]'
    elif split == 'test':  # 2700 records
        ds_split = 'train[90%:]'
    else:
        raise Exception("unexpected split", split)

    def to_float(x, y):
        x = tf.cast(x, tf.float32) / 255
        return x, y

    dataset = (tfds.load('eurosat/rgb', split=ds_split, shuffle_files=True,
                         batch_size=batch_size, as_supervised=True)
               .map(to_float)
               .prefetch(tf.data.experimental.AUTOTUNE))
    return tfds.as_numpy(dataset)


# ds = dataset('test', batch_size=4)
# for imgs, labels in ds:
#     print("imgs", imgs.shape)
#     print(imgs[0])
#     print("labels", labels)
#     break
