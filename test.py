import objax
import models
import argparse
import sys
import data
import util

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-models', type=int, default=1)
parser.add_argument('--dense-kernel-size', type=int, default=16)
parser.add_argument('--saved-model', type=str, required=True)
parser.add_argument('--split', type=str, default='test')
opts = parser.parse_args()
print(opts, file=sys.stderr)

if opts.split not in ['validate', 'test']:
    raise Exception()

# restore model
if opts.num_models == 1:
    net = models.NonEnsembleNet(num_classes=10,
                                dense_kernel_size=opts.dense_kernel_size,
                                seed=0)
else:
    net = models.EnsembleNet(num_models=opts.num_models,
                             num_classes=10,
                             dense_kernel_size=opts.dense_kernel_size,
                             seed=0)
objax.io.load_var_collection(opts.saved_model, net.vars())

# read entire dataset
if opts.split == 'validate':
    imgs, labels = data.validation_dataset()
else:
    imgs, labels = data.test_dataset()

# final validation metrics
accuracy = util.accuracy(net.predict(imgs), labels)
print("accuracy", opts.split, accuracy)
