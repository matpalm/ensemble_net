import os
import numpy as np
import datetime
import time


def DTS():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_dir_exists_for_file(fname):
    ensure_dir_exists(os.path.dirname(fname))


class EarlyStopping(object):
    def __init__(self, patience=3, burn_in=3, max_runtime=None):
        self.original_patience = patience
        self.patience = patience
        self.burn_in = burn_in
        self.lowest_value = None
        self.decided_to_stop = False
        if max_runtime is not None:
            self.exit_time = time.time() + max_runtime
        else:
            self.exit_time = None

    def should_stop(self, value):
        # if we've already decided to stop then return True immediately
        if self.decided_to_stop:
            return True

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
            self.lowest_value = value

        # check if we've made an improvement; if so reset patience and record
        # new lowest
        made_improvement = value < self.lowest_value
        if made_improvement:
            self.patience = self.original_patience
            self.lowest_value = value
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


def accuracy(predictions, labels):
    num_correct = np.equal(predictions, labels).sum()
    num_total = len(predictions)
    return num_correct / num_total
