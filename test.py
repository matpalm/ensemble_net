import models
import argparse
import sys
import data
import util
import pickle
from jax import vmap, jit
import jax.numpy as jnp

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--models-per-device', type=int, default=2)
parser.add_argument('--max-conv-size', type=int, default=64)
parser.add_argument('--dense-kernel-size', type=int, default=16)
parser.add_argument('--saved-params', type=str, required=True)
opts = parser.parse_args()

with open(opts.saved_params, "rb") as f:
    params = pickle.load(f)

model = models.build_model(opts)

# plumb batch dimension for models_per_device
all_models_apply = vmap(model.apply, in_axes=(0, None))

# plumb batch dimension for num_devices
all_models_apply = vmap(all_models_apply, in_axes=(0, None))

num_classes = 10

# TODO: move this into util so can run during training too
#       ( as well as being run against validation and test )


@jit
def predict(params, imgs):
    logits = all_models_apply(params, imgs)
    batch_size = logits.shape[-2]
    logits = logits.reshape((-1, batch_size, num_classes))  # (M, B, 10)
    ensembled_logits = jnp.sum(logits, axis=0)              # (B, 10)
    predictions = jnp.argmax(ensembled_logits, axis=-1)     # (B)
    return predictions


num_correct = 0
num_total = 0
dataset = data.validation_dataset(batch_size=64)
for imgs, labels in dataset:
    predictions = predict(params, imgs)
    num_correct += jnp.sum(predictions == labels)
    num_total += len(labels)

accuracy = num_correct / num_total
print(num_correct, num_total)

# # restore from save
# objax.io.load_var_collection(opts.saved_model, net.vars())

# # check against validation set
# accuracy = util.accuracy(net, data.validation_dataset(batch_size=128))
# print("validation accuracy %0.3f" % accuracy)

# # check against test set
# accuracy = util.accuracy(net, data.test_dataset(batch_size=128))
# print("test accuracy %0.3f" % accuracy)
