import argparse
import models
import numpy as np
import jax.numpy as jnp
import objax
import jax
import data
import util
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys


def cm_plot(cm):
    labels = ['Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway',
              'Industrial Buildings', 'Pasture', 'Permanent Crop',
              'Residential Buildings', 'River', 'Sea & Lake']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels)
    return disp.plot(include_values=True,
                     cmap='viridis', ax=None, xticks_rotation='vertical',
                     values_format=None)


def save_plot(y_true, y_pred, title, fname):
    plot = cm_plot(confusion_matrix(y_true, y_pred))
    plot.figure_.suptitle(title)
    plot.figure_.savefig(fname, bbox_inches='tight', transparent=True)


def save_sub_model_plots(y_true, logits, num_models, title_template,
                         fname_template):
    for m in range(num_models):
        y_pred = jnp.argmax(logits[m], axis=-1)
        num_correct = np.equal(y_pred, y_true).sum()
        num_total = len(y_true)
        print("model %d accuracy %0.3f" % (m, float(num_correct / num_total)))
        save_plot(y_true, y_pred, title_template % m, fname_template % m)


def print_validation_test_accuracy(net):
    print("validation %0.3f" % util.accuracy(
        net, data.validation_dataset(batch_size=100)))
    print("test %0.3f" % util.accuracy(net, data.test_dataset(batch_size=100)))


def logits_and_y_true_for_test_set(net, num_models):
    logits = []
    y_true = []
    for imgs, labels in data.test_dataset(batch_size=100):
        logits.append(net.logits(imgs, single_result=False,
                                 model_dropout=False))
        y_true.extend(labels)
    logits = jnp.stack(logits)                       # (27, M, 100, 10)
    logits = logits.transpose((1, 0, 2, 3))          # (M, 27, 100, 10)
    logits = logits.reshape((num_models, 2700, 10))  # (M, 2700, 10)
    return logits, y_true


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-models', type=int, default=1)
parser.add_argument('--max-conv-size', type=int)
parser.add_argument('--dense-kernel-size', type=int)
parser.add_argument('--ckpt-file', type=str)
opts = parser.parse_args()
print(opts, file=sys.stderr)

net = models.EnsembleNet(num_models=opts.num_models,
                         num_classes=10,
                         max_conv_size=opts.max_conv_size,
                         dense_kernel_size=opts.dense_kernel_size,
                         seed=0)
objax.io.load_var_collection(opts.ckpt_file, net.vars())

logits, y_true = logits_and_y_true_for_test_set(net, opts.num_models)

y_pred = jnp.argmax(logits.sum(axis=0), axis=-1)

print_validation_test_accuracy(net)

save_plot(y_true, y_pred, "ensemble", "cm.ensemble.png")
save_sub_model_plots(y_true, logits, opts.num_models,
                     "sub model %d", "cm.model_%d.png")
