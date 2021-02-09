import jax.numpy as jnp
from jax.nn import gelu
import haiku as hk
from functools import partial


def global_spatial_mean_pooling(x):
    return jnp.mean(x, axis=(1, 2))


def model(x, dense_kernel_size=64, max_conv_size=256, num_classes=10):
    layers = []
    for c in [32, 64, 128, 256]:
        layers.append(hk.Conv2D(output_channels=min(c, max_conv_size),
                                kernel_shape=3, stride=2))
        layers.append(gelu)
    layers += [global_spatial_mean_pooling,
               hk.Linear(dense_kernel_size),
               gelu,
               hk.Linear(num_classes)]
    return hk.Sequential(layers)(x)


def build_model(opts):
    m = partial(model, max_conv_size=opts.max_conv_size,
                dense_kernel_size=opts.dense_kernel_size)
    return hk.without_apply_rng(hk.transform(m))
