import os


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_dir_exists_for_file(fname):
    ensure_dir_exists(os.path.dirname(fname))


class EarlyStopping(object):
    def __init__(self, patience=3, burn_in=3):
        self.original_patience = patience
        self.patience = patience
        self.burn_in = burn_in
        self.lowest_value = None
        self.already_decided_to_stop = False

    def should_stop(self, value):
        # print("CHECK! value", value, "lowest_value",  self.lowest_value,
        #       "burn_in", self.burn_in, "patience", self.patience)

        # if we've already decided to stop then return True immediately
        if self.already_decided_to_stop:
            #print("LATCH!, already decided stop")
            return True

        # ignore first burn_in iterations
        if self.burn_in > 0:
            #print("burning in!")
            self.burn_in -= 1
            return False

        # take very first value we see as the lowest
        if self.lowest_value is None:
            #print("first value is initial lowest")
            self.lowest_value = value

        # check if we've made an improvement; if so reset patience and record
        # new lowest
        made_improvement = value < self.lowest_value
        if made_improvement:
            #print("made improvement!")
            self.patience = self.original_patience
            self.lowest_value = value
            return False

        # if no improvement made reduce patience. if no more patience exit.
        self.patience -= 1
        if self.patience == 0:
            #print("NO MORE PATIENCE!")
            self.already_decided_to_stop = True
            return True
        else:
            return False
