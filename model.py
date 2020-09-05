import jax
import jax.numpy as jnp
from jax import random, lax, vmap
from jax.nn.initializers import glorot_normal, he_normal
from jax.nn.functions import gelu
from functools import partial
import objax
from objax.variable import TrainVar


def _conv_layer(stride, activation, inp, kernel, bias):
    no_dilation = (1, 1)
    some_height_width = 10  # values don't matter; just shape of input
    input_shape = (1, some_height_width, some_height_width, 3)
    kernel_shape = (3, 3, 1, 1)
    input_kernel_output = ('NHWC', 'HWIO', 'NHWC')
    conv_dimension_numbers = lax.conv_dimension_numbers(input_shape,
                                                        kernel_shape,
                                                        input_kernel_output)
    block = lax.conv_general_dilated(inp, kernel, (stride, stride),
                                     'VALID', no_dilation, no_dilation,
                                     conv_dimension_numbers)
    if bias is not None:
        block += bias
    if activation:
        block = activation(block)
    return block


def _dense_layer(activation, inp, kernel, bias):
    block = jnp.dot(inp, kernel) + bias
    if activation:
        block = activation(block)
    return block

# def _conv_block_without_bias(stride, with_non_linearity, inp, kernel):
#     # the need for this method feels a bit clunky :/ is there a better
#     # way to vmap with the None?
#     return _conv_block(stride, with_non_linearity, inp, kernel, None)


class NonEnsembleNet(objax.Module):

    def __init__(self, num_classes, dense_kernel_size=32, seed=0):

        key = random.PRNGKey(seed)
        subkeys = random.split(key, 8)

        # conv stack kernels and biases
        self.conv_kernels = objax.ModuleList()
        self.conv_biases = objax.ModuleList()
        input_channels = 3
        for i, output_channels in enumerate([32, 64, 64, 64]):
            self.conv_kernels.append(TrainVar(he_normal()(
                subkeys[i], (3, 3, input_channels, output_channels))))
            self.conv_biases.append(TrainVar(jnp.zeros((output_channels))))
            input_channels = output_channels

        # dense layer kernel and bias
        self.dense_kernel = TrainVar(he_normal()(
            subkeys[6], (output_channels, dense_kernel_size)))
        self.dense_bias = TrainVar(jnp.zeros((dense_kernel_size)))

        # classifier layer kernel and bias
        self.logits_kernel = TrainVar(glorot_normal()(
            subkeys[6], (dense_kernel_size, num_classes)))
        self.logits_bias = TrainVar(jnp.zeros((num_classes)))

    def logits(self, inp):
        # conv stack -> (B, 3, 3, 64)
        y = inp
        for kernel, bias in zip(self.conv_kernels, self.conv_biases):
            y = _conv_layer(2, gelu, y, kernel.value, bias.value)

        # global spatial pooling -> (B, 64)
        y = jnp.mean(y, axis=(1, 2))

        # dense layer with non linearity -> (B, 32)
        y = _dense_layer(gelu, y, self.dense_kernel.value,
                         self.dense_bias.value)

        # dense layer with no activation to number classes -> (B, num_classes)
        logits = _dense_layer(
            None, y, self.logits_kernel.value, self.logits_bias.value)

        return logits

    def predict(self, inp):
        return jax.nn.softmax(self.logits(inp))
