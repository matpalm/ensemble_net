import os
import numpy as np
import datetime
import time
import data
from objax.functional.loss import cross_entropy_logits_sparse
import jax.numpy as jnp


def DTS():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_dir_exists_for_file(fname):
    ensure_dir_exists(os.path.dirname(fname))


class EarlyStopping(object):
    def __init__(self, patience=3, burn_in=5, max_runtime=None,
                 smoothing=0.0):
        # smoothing = 0.0 => no smoothing

        self.original_patience = patience
        self.patience = patience
        self.burn_in = burn_in
        self.lowest_value = None

        self.decided_to_stop = False
        if max_runtime is not None:
            self.exit_time = time.time() + max_runtime
        else:
            self.exit_time = None

        if smoothing < 0.0 or smoothing > 1.0:
            raise Exception("invalid smoothing value %s" % smoothing)
        self.smoothing = 1.0 - smoothing
        self.smoothed_value = None

    def should_stop(self, value):
        # if we've already decided to stop then return True immediately
        if self.decided_to_stop:
            return True

        # calc smoothed value
        if self.smoothed_value is None:
            self.smoothed_value = value
        else:
            self.smoothed_value += self.smoothing * \
                (value - self.smoothed_value)

        # run taken too long?
        if self.exit_time is not None:
            if time.time() > self.exit_time:
                self.decided_to_stop = True
                return True

        # ignore first burn_in iterations
        if self.burn_in > 0:
            self.burn_in -= 1
            return False

        # take very first value we see as the lowest
        if self.lowest_value is None:
            self.lowest_value = self.smoothed_value

        # check if we've made an improvement; if so reset patience and record
        # new lowest
        made_improvement = self.smoothed_value < self.lowest_value
        if made_improvement:
            self.patience = self.original_patience
            self.lowest_value = self.smoothed_value
            return False

        # if no improvement made reduce patience. if no more patience exit.
        self.patience -= 1
        if self.patience == 0:
            self.decided_to_stop = True
            return True
        else:
            return False

    def stopped(self):
        return self.decided_to_stop


def mean_loss(net, dataset):
    # TODO: could go to NonEnsembleNet/EnsembleNet base class
    losses_total = 0
    num_losses = 0
    for imgs, labels in dataset:
        logits = net.logits(imgs, single_result=True, logits_dropout=False)
        losses = cross_entropy_logits_sparse(logits, labels)
        losses_total += jnp.sum(losses)
        num_losses += len(losses)
    return losses_total / num_losses


def accuracy(net, dataset):
    # TODO: could go to NonEnsembleNet/EnsembleNet base class
    y_pred = []
    y_true = []
    for imgs, labels in dataset:
        y_pred.extend(net.predict(imgs, single_result=True))
        y_true.extend(labels)
    num_correct = np.equal(y_pred, y_true).sum()
    num_total = len(y_pred)
    return num_correct / num_total
