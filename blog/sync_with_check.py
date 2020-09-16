#!/usr/bin/env python3
import subprocess
import re
import os
import shutil


def mv_run_matching_hash(run_hash):
    for run in os.listdir("wandb"):
        #print("run?", run)
        if run_hash in run:
            #print("HASH", run_hash, "matches", run)
            srcdir = f"/home/mat/dev/ensemble_net/wandb/{run}"
            destdir = f"/tmp/{run}"
            print(f"ERROR: mv [{srcdir}] [{destdir}]")
            shutil.move(srcdir, destdir)
            return


def check_at_least_one_run():
    for subdir in os.listdir("wandb"):
        if subdir.startswith("run"):
            return
    print("NO RUNS!")
    exit()


check_at_least_one_run()

try:
    result = subprocess.run(['wandb', 'sync'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
except Exception as e:
    print("non zero exit?", str(e))

# for line in result.stdout.decode('utf-8').split("\n"):
#    print("stdout [", line, "]")


syncing = None
for line in result.stderr.decode('utf-8').split("\n"):

    print(f"PROCESS [{line}]")

    m = re.match(".*No history or tfevents files found.*")
    if m is not None:
        print("DONE!")
        exit()

    m = re.match(".*Syncing (.*) to.*", line)
    if m is not None:
        syncing = m.group(1)
        print(">SYNCING [%s]" % syncing)
        continue

    if re.match(".*Finished!.*", line) is not None:
        if syncing is None:
            raise Exception("Finished, but not in a sycn?")
        srcdir = syncing
        destdir = "/tmp/%s" % syncing.split("/")[-1]
        print(f">SYNCING  mv [{srcdir}] [{destdir}]")
        shutil.move(srcdir, destdir)
        syncing = None
        continue

    m = re.match(
        ".*ERROR Error while calling.*ensemble_net/(.*) was previous.*", line)
    if m is not None:
        run_hash = m.group(1)
        #print("ERROR", run_hash)
        mv_run_matching_hash(run_hash)
        continue
