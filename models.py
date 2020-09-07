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


class NonEnsembleNet(objax.Module):

    def __init__(self, num_classes, dense_kernel_size=32, seed=0):
        key = random.PRNGKey(seed)
        subkeys = random.split(key, 8)

        # conv stack kernels and biases
        self.conv_kernels = objax.ModuleList()
        self.conv_biases = objax.ModuleList()
        input_channels = 3
        for i, output_channels in enumerate([32, 64, 128, 256]):
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

    def logits(self, inp, single_result):
        """return logits over inputs
        Args:
          inp: input images  (B, HW, HW, 3)
        Returns:
          logit values for input images  (B, C)
        """

        # conv stack -> (B, 3, 3, 256)
        y = inp
        for kernel, bias in zip(self.conv_kernels, self.conv_biases):
            y = _conv_layer(2, gelu, y, kernel.value, bias.value)

        # global spatial pooling -> (B, 256)
        y = jnp.mean(y, axis=(1, 2))

        # dense layer with non linearity -> (B, DKS)
        y = _dense_layer(gelu, y, self.dense_kernel.value,
                         self.dense_bias.value)

        # dense layer with no activation to number classes -> (B, C)
        logits = _dense_layer(
            None, y, self.logits_kernel.value, self.logits_bias.value)

        return logits

    def predict_proba(self, inp, single_result):
        """return prediction probabilities. i.e. softmax over logits.
        Args:
          inp: input images  (B, HW, HW, 3)
        Returns:
          softmax values for input images  (B, C)
        """
        return jax.nn.softmax(self.logits(inp, single_result), axis=-1)

    def predict(self, inp, single_result):
        """return class predictions. i.e. argmax over logits.
        Args:
          inp: input images  (B, HW, HW, 3)
        Returns:
          prediction classes for input images  (B,)
        """
        return jnp.argmax(self.logits(inp, single_result), axis=-1)


class EnsembleNet(objax.Module):

    def __init__(self, num_models, num_classes, dense_kernel_size=32, seed=0):
        self.num_models = num_models

        key = random.PRNGKey(seed)
        subkeys = random.split(key, 8)

        # conv stack kernels and biases
        self.conv_kernels = objax.ModuleList()
        self.conv_biases = objax.ModuleList()
        input_channels = 3
        for i, output_channels in enumerate([32, 64, 64, 64]):
            self.conv_kernels.append(TrainVar(he_normal()(
                subkeys[i], (num_models, 3, 3, input_channels,
                             output_channels))))
            self.conv_biases.append(
                TrainVar(jnp.zeros((num_models, output_channels))))
            input_channels = output_channels

        # dense layer kernel and bias
        self.dense_kernel = TrainVar(he_normal()(
            subkeys[6], (num_models, output_channels, dense_kernel_size)))
        self.dense_bias = TrainVar(jnp.zeros((num_models, dense_kernel_size)))

        # classifier layer kernel and bias
        self.logits_kernel = TrainVar(glorot_normal()(
            subkeys[6], (num_models, dense_kernel_size, num_classes)))
        self.logits_bias = TrainVar(jnp.zeros((num_models, num_classes)))

    def logits(self, inp, single_result):
        """return logits over inputs.
        Args:
          inp: input images. either (B, HW, HW, 3) in which case all models
            will get the same images or (M, B, HW, HW, 3) in which case each
            model will get a different image.
        Returns:
          logit values for input images. either (B, C) if in single_result mode
          or (M, B, C) otherwise.
        Raises:
          Exception: if input images are (M, B, HW, HW, 3) and in single_result
                     mode.
        """

        if len(inp.shape) == 4:
            # inp (B, HW, HW, 3)
            # apply first convolution as vmap against just inp
            y = vmap(partial(_conv_layer, 2, gelu, inp))(
                self.conv_kernels[0].value, self.conv_biases[0].value)
        elif len(inp.shape) == 5:
            if single_result:
                raise Exception("self.single_result=True not valid when passed"
                                " an image per model")
            if inp.shape[0] != self.num_models:
                raise Exception("when passing (M, B, HW, HW, 3) the leading"
                                " dimension needs to match the number of models"
                                " in the ensemble.")
            # inp (M, B, HW, HW, 3)
            # apply all convolutions, including first, as vmap against both y
            # and kernel, bias
            y = vmap(partial(_conv_layer, 2, gelu))(
                inp, self.conv_kernels[0].value, self.conv_biases[0].value)
        else:
            raise Exception("unexpected input shape")

        # y now (M, B, HW/2 HW/2, 32)

        # rest of the convolution stack can be applied as vmap against both y
        # and conv kernels, biases. final result is (M, B, 3, 3, 64)
        for kernel, bias in zip(self.conv_kernels[1:], self.conv_biases[1:]):
            y = vmap(partial(_conv_layer, 2, gelu))(
                y, kernel.value, bias.value)

        # global spatial pooling. (M, B, 64)
        y = jnp.mean(y, axis=(2, 3))

        # dense layer with non linearity. (M, B, 32)
        y = vmap(partial(_dense_layer, gelu))(
            y, self.dense_kernel.value, self.dense_bias.value)

        # dense layer with no activation to number classes.
        # (M, B, num_classes)
        logits = vmap(partial(_dense_layer, None))(
            y, self.logits_kernel.value, self.logits_bias.value)

        if single_result:
            # sum logits over models to represent single ensemble result
            # (B, num_classes)
            logits = jnp.sum(logits, axis=0)

        return logits

    def predict_proba(self, inp, single_result):
        """return prediction probabilities. i.e. softmax over logits.
        Args:
          inp: input images. either (B, HW, HW, 3) in which case all models
            will get the same images or (M, B, HW, HW, 3) in which case each
            model will get a different image.
        Returns:
          softmax values for input images. either (B, C) if in single_result
          mode or (M, B, C) otherwise.
        Raises:
          Exception: if input images are (M, B, HW, HW, 3) and in single_result
                     mode.
        """

        return jax.nn.softmax(self.logits(inp, single_result), axis=-1)

    def predict(self, inp, single_result):
        """return class predictions. i.e. argmax over logits.
        Args:
          inp: input images. either (B, HW, HW, 3) in which case all models
            will get the same images or (M, B, HW, HW, 3) in which case each
            model will get a different image.
        Returns:
          prediction classes for input images. either (B,) if in single_result
          mode or (M, B) otherwise.
        Raises:
          Exception: if input images are (M, B, HW, HW, 3) and in single_result
                     mode.
        """

        return jnp.argmax(self.logits(inp, single_result), axis=-1)
