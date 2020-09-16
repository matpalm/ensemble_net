#!/usr/bin/env python3
# should have done this during training, but whatever

import sys
import re

mins = {}
configs = {}
config = None

for line in sys.stdin.readlines():
    line = line.strip()

    m = re.match("^starting \[(.*)", line)
    if m:
        config = m.group(1)
        continue

    m = re.match("^starting run (.*)", line)
    if m:
        assert config is not None
        run = m.group(1)
        configs[run] = config
        config = None
        continue

    m = re.match("(.*) epoch (.*) validation_loss (.*)", line)
    if m:
        run, epoch, loss = m.groups()
        epoch = int(epoch)
        loss = float(loss)

        take = False
        if run in mins:
            take = loss < mins[run][1]
        else:
            take = True

        if take:
            mins[run] = epoch, loss


for run in sorted(mins.keys()):
    print(run, mins[run], configs[run])
