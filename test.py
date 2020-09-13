import objax
import models
import argparse
import sys
import data
import util

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-models', type=int, default=1)
parser.add_argument('--max-conv-size', type=int, default=64)
parser.add_argument('--dense-kernel-size', type=int, default=16)
parser.add_argument('--saved-model', type=str, required=True)
opts = parser.parse_args()

# construct model
# TODO: would be better to be able to load from save without this config
if opts.num_models == 1:
    net = models.NonEnsembleNet(num_classes=10,
                                max_conv_size=opts.max_conv_size,
                                dense_kernel_size=opts.dense_kernel_size,
                                seed=0)
else:
    net = models.EnsembleNet(num_models=opts.num_models,
                             num_classes=10,
                             max_conv_size=opts.max_conv_size,
                             dense_kernel_size=opts.dense_kernel_size,
                             seed=0)

# restore from save
objax.io.load_var_collection(opts.saved_model, net.vars())

# check against validation set
accuracy = util.accuracy(net, data.validation_dataset(batch_size=128))
print("validation accuracy %0.3f" % accuracy)

# check against test set
accuracy = util.accuracy(net, data.test_dataset(batch_size=128))
print("test accuracy %0.3f" % accuracy)
