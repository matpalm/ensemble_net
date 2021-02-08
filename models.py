# import jax
# import jax.numpy as jnp
# from jax import random, lax, vmap
# from jax.nn.initializers import glorot_normal, he_normal
# from jax.nn.functions import gelu
# from functools import partial
# import objax
# from objax.variable import TrainVar, StateVar

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

# def _conv_layer(stride, activation, inp, kernel, bias):
#     no_dilation = (1, 1)
#     some_height_width = 10  # values don't matter; just shape of input
#     input_shape = (1, some_height_width, some_height_width, 3)
#     kernel_shape = (3, 3, 1, 1)
#     input_kernel_output = ('NHWC', 'HWIO', 'NHWC')
#     conv_dimension_numbers = lax.conv_dimension_numbers(input_shape,
#                                                         kernel_shape,
#                                                         input_kernel_output)
#     block = lax.conv_general_dilated(inp, kernel, (stride, stride),
#                                      'VALID', no_dilation, no_dilation,
#                                      conv_dimension_numbers)
#     if bias is not None:
#         block += bias
#     if activation:
#         block = activation(block)
#     return block


# def _dense_layer(activation, inp, kernel, bias):
#     block = jnp.dot(inp, kernel) + bias
#     if activation:
#         block = activation(block)
#     return block

# # TODO: introduce basecase for Ensemble & NonEnsemble to avoid
# #       clumsy single_result & model_dropout on NonEnsemble


# class NonEnsembleNet(objax.Module):

#     def __init__(self, num_classes, max_conv_size=64, dense_kernel_size=32,
#                  seed=0):
#         key = random.PRNGKey(seed)
#         subkeys = random.split(key, 6)

#         # conv stack kernels and biases
#         self.conv_kernels = objax.ModuleList()
#         self.conv_biases = objax.ModuleList()
#         input_channels = 3
#         for i, output_channels in enumerate([32, 64, 128, 256]):
#             output_channels = min(output_channels, max_conv_size)
#             self.conv_kernels.append(TrainVar(he_normal()(
#                 subkeys[i], (3, 3, input_channels, output_channels))))
#             self.conv_biases.append(TrainVar(jnp.zeros((output_channels))))
#             input_channels = output_channels

#         # dense layer kernel and bias
#         self.dense_kernel = TrainVar(he_normal()(
#             subkeys[4], (output_channels, dense_kernel_size)))
#         self.dense_bias = TrainVar(jnp.zeros((dense_kernel_size)))

#         # classifier layer kernel and bias
#         self.logits_kernel = TrainVar(glorot_normal()(
#             subkeys[5], (dense_kernel_size, num_classes)))
#         self.logits_bias = TrainVar(jnp.zeros((num_classes)))

#     def logits(self, inp, single_result=None, model_dropout=None):
#         """return logits over inputs
#         Args:
#           inp: input images  (B, HW, HW, 3)
#           single_result: clumsily ignore for NonEnsembleNet :/
#           model_dropout: clumsily ignore for NonEnsembleNet :/
#         Returns:
#           logit values for input images  (B, C)
#         """

#         # conv stack -> (B, 3, 3, MCS)   MaxConvSize
#         y = inp
#         for kernel, bias in zip(self.conv_kernels, self.conv_biases):
#             y = _conv_layer(2, gelu, y, kernel.value, bias.value)

#         # global spatial pooling -> (B, MCS)
#         y = jnp.mean(y, axis=(1, 2))

#         # dense layer with non linearity -> (B, DKS)  DenseKernelSize
#         y = _dense_layer(gelu, y, self.dense_kernel.value,
#                          self.dense_bias.value)

#         # dense layer with no activation to number classes -> (B, C)
#         logits = _dense_layer(
#             None, y, self.logits_kernel.value, self.logits_bias.value)

#         return logits

#     def predict(self, inp, single_result=None):
#         """return class predictions. i.e. argmax over logits.
#         Args:
#           inp: input images  (B, HW, HW, 3)
#           single_result: clumsily ignore for NonEnsembleNet :/
#         Returns:
#           prediction classes for input images  (B,)
#         """
#         return jnp.argmax(self.logits(inp, single_result), axis=-1)


# class EnsembleNet(objax.Module):

#     def __init__(self, num_models, num_classes, max_conv_size=64,
#                  dense_kernel_size=32, seed=0):
#         self.num_models = num_models

#         key = random.PRNGKey(seed)
#         subkeys = random.split(key, 7)

#         # conv stack kernels and biases
#         self.conv_kernels = objax.ModuleList()
#         self.conv_biases = objax.ModuleList()
#         input_channels = 3
#         for i, output_channels in enumerate([32, 64, 128, 256]):
#             output_channels = min(output_channels, max_conv_size)
#             self.conv_kernels.append(TrainVar(he_normal()(
#                 subkeys[i], (num_models, 3, 3, input_channels,
#                              output_channels))))
#             self.conv_biases.append(
#                 TrainVar(jnp.zeros((num_models, output_channels))))
#             input_channels = output_channels

#         # dense layer kernel and bias
#         self.dense_kernel = TrainVar(he_normal()(
#             subkeys[4], (num_models, output_channels, dense_kernel_size)))
#         self.dense_bias = TrainVar(jnp.zeros((num_models, dense_kernel_size)))

#         # classifier layer kernel and bias
#         self.logits_kernel = TrainVar(glorot_normal()(
#             subkeys[5], (num_models, dense_kernel_size, num_classes)))
#         self.logits_bias = TrainVar(jnp.zeros((num_models, num_classes)))

#         # create a rng key generator for doing dropout each call to logits
#         # (as required)
#         self.dropout_key = objax.random.Generator(seed)

#     def logits(self, inp, single_result, model_dropout):
#         """return logits over inputs.
#         Args:
#           inp: input images. either (B, HW, HW, 3) in which case all models
#             will get the same images or (M, B, HW, HW, 3) in which case each
#             model will get a different image.
#           single_result: if true return single logits value for ensemble.
#             otherwise return logits for each sub model.
#           model_dropout: if true then only run 50% of the models during
#             training.
#         Returns:
#           logit values for input images. either (B, C) if in single_result mode
#           or (M, B, C) otherwise.
#         Raises:
#           Exception: if input images are (M, B, HW, HW, 3) and in single_result
#                      mode.
#         """

#         if not model_dropout:
#             # if we are not doing model dropout then just take all the model
#             # variables as they are. these will be of shape (M, ...)
#             conv_kernels = [k.value for k in self.conv_kernels]
#             conv_biases = [b.value for b in self.conv_biases]
#             dense_kernel = self.dense_kernel.value
#             dense_bias = self.dense_bias.value
#             logits_kernel = self.logits_kernel.value
#             logits_bias = self.logits_bias.value
#         else:
#             # but if we are doing model dropout then we take half the models
#             # by first 1) picking idxs that represents a random half of the
#             # models ...
#             idxs = jnp.arange(self.num_models)
#             idxs = jax.random.permutation(self.dropout_key(), idxs)
#             idxs = idxs[:(self.num_models//2)]
#             # ... and then 2) slicing the variables out. all these will be of
#             # shape (M/2, ...)
#             conv_kernels = [k.value[idxs] for k in self.conv_kernels]
#             conv_biases = [b.value[idxs] for b in self.conv_biases]
#             dense_kernel = self.dense_kernel.value[idxs]
#             dense_bias = self.dense_bias.value[idxs]
#             logits_kernel = self.logits_kernel.value[idxs]
#             logits_bias = self.logits_bias.value[idxs]

#         if len(inp.shape) == 4:
#             # single_input mode; inp (B, HW, HW, 3)
#             # apply first convolution as vmap against just inp
#             y = vmap(partial(_conv_layer, 2, gelu, inp))(conv_kernels[0],
#                                                          conv_biases[0])
#         elif len(inp.shape) == 5:
#             # multi_input mode; inp (M, B, HW, HW, 3)
#             if single_result:
#                 raise Exception("self.single_result=True not valid when passed"
#                                 " an image per model")
#             if inp.shape[0] != self.num_models:
#                 raise Exception("when passing (M, B, HW, HW, 3) the leading"
#                                 " dimension needs to match the number of models"
#                                 " in the ensemble.")
#             # apply all convolutions, including first, as vmap against both y
#             # and kernel, bias
#             y = vmap(partial(_conv_layer, 2, gelu))(inp, conv_kernels[0],
#                                                     conv_biases[0])
#         else:
#             raise Exception("unexpected input shape")

#         # y now (M, B, HW/2 HW/2, 32)

#         # rest of the convolution stack can be applied as vmap against both y
#         # and conv kernels, biases.
#         # final result is (M, B, 3, 3, 256|max_conv_size)
#         for kernel, bias in zip(conv_kernels[1:], conv_biases[1:]):
#             y = vmap(partial(_conv_layer, 2, gelu))(y, kernel, bias)

#         # global spatial pooling. (M, B, dense_kernel_size)
#         y = jnp.mean(y, axis=(2, 3))

#         # dense layer with non linearity. (M, B, dense_kernel_size)
#         y = vmap(partial(_dense_layer, gelu))(y, dense_kernel, dense_bias)

#         # dense layer with no activation to number classes.
#         # (M, B, num_classes)
#         logits = vmap(partial(_dense_layer, None))(y, logits_kernel,
#                                                    logits_bias)

#         # if single result sum logits over models to represent single
#         # ensemble result (B, num_classes)
#         if single_result:
#             logits = jnp.sum(logits, axis=0)

#         return logits

#     def predict(self, inp, single_result):
#         """return class predictions. i.e. argmax over logits.
#         Args:
#           inp: input images. either (B, HW, HW, 3) in which case all models
#             will get the same images or (M, B, HW, HW, 3) in which case each
#             model will get a different image.
#           single_result: if true a single prediction for ensemble is returned.
#             otherwise return predictions per sub model.
#         Returns:
#           prediction classes for input images. either (B,) if in single_result
#           mode or (M, B) otherwise.
#         Raises:
#           Exception: if input images are (M, B, HW, HW, 3) and in single_result
#                      mode.
#         """

#         return jnp.argmax(self.logits(inp, single_result, False), axis=-1)
