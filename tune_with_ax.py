
import argparse
import train
from ax.service.ax_client import AxClient
import time
import sys
import util as u
import random
import json
import tensorflow as tf

# tf.config.experimental.set_visible_devices([], "GPU")

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--group', type=str, default=None,
                    help='wandb group. if none, no logging')
parser.add_argument('--mode', type=str, required=True,
                    help='mode; one of siso, simo, simo_ld or mimo')
parser.add_argument('--num-models', type=int, default=1)
parser.add_argument('--run-time-sec', type=int, default=60*10)
parser.add_argument('--epochs', type=int, default=60)

cmd_line_opts = parser.parse_args()
print(cmd_line_opts, file=sys.stderr)

if cmd_line_opts.mode not in ['siso', 'simo', 'simo_ld', 'mimo']:
    raise Exception("invalid --mode")

# note: if running on display GPU you probably want to run set env var
# something like XLA_PYTHON_CLIENT_MEM_FRACTION=.8 to allow jobs tuned too
# large to faily cleanly with OOM

# we tune for 4 major configuration combos; see train.py for more info.

# a) SISO --input-mode=single --num-models=1
#    this is the baseline non ensemble config.
#
# b) SIMO --input-mode=single --num-models=M
#    models single set of inputs and labels. ensemble outputs are summed at
#    logits summed to produce single output.
#
# c) SIMO_LD --input-mode=single --num-models=M
#    models single set of inputs and labels. ensemble outputs are summed at
#    logits summed, after dropout, to produce single output.
#
# d) MIMO --input-mode=multiple --num-models=M
#    multiple inputs (with multiple labels) going through multiple models.
#    loss is still averaged over all models though.

ax_params = [
    {
        "name": "max_conv_size",
        "type": "range",
        "bounds": [8, 256],
    },
    {
        "name": "dense_kernel_size",
        "type": "range",
        "bounds": [8, 128],
    },
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [1e-4, 1e-1],
        "log_scale": True,
    },
    # {
    #     "name": "batch_size",
    #     "type": "choice",
    #     "values": [32, 64],
    # },
]

# if cmd_line_opts.mode in ['simo', 'mimo']:
#     ax_params.append({
#         "name": "num_models",
#         "type": "range",
#         "bounds": [2, 8],
#     })

ax = AxClient()
ax.create_experiment(
    name="ensemble_net_tuning",
    parameters=ax_params,
    objective_name="final_loss",
    minimize=True,
)

u.ensure_dir_exists("logs/%s" % cmd_line_opts.group)
log = open("logs/%s/ax_trials.tsv" % cmd_line_opts.group, "w")
print("trial_index\tparameters\truntime\tfinal_loss", file=log)

end_time = time.time() + cmd_line_opts.run_time_sec

while time.time() < end_time:

    parameters, trial_index = ax.get_next_trial()
    log_record = [trial_index, json.dumps(parameters)]
    print("starting", log_record)

    class Opts(object):
        pass
    opts = Opts()

    opts.group = cmd_line_opts.group
    opts.seed = random.randint(0, 1e9)

    if cmd_line_opts.mode == 'siso':
        opts.input_mode = 'single'
        opts.num_models = 1
        opts.logits_dropout = False  # N/A for multi_input
    elif cmd_line_opts.mode == 'simo':
        opts.input_mode = 'single'
        opts.num_models = cmd_line_opts.num_models
        opts.logits_dropout = False  # not yet under tuning
    elif cmd_line_opts.mode == 'simo_ld':
        opts.input_mode = 'single'
        opts.num_models = cmd_line_opts.num_models
        opts.logits_dropout = True  # not yet under tuning
    else:  # mimo
        opts.input_mode = 'multiple'
        opts.num_models = cmd_line_opts.num_models
        opts.logits_dropout = False  # N/A for multi_input

    opts.max_conv_size = parameters['max_conv_size']
    opts.dense_kernel_size = parameters['dense_kernel_size']
    opts.batch_size = 64  # parameters['batch_size']
    opts.learning_rate = parameters['learning_rate']
    opts.epochs = cmd_line_opts.epochs  # max to run, we also use early stopping

    # run
    start_time = time.time()
    # final_loss = train.train_in_subprocess(opts)
    final_loss = train.train(opts)
    log_record.append(time.time() - start_time)
    log_record.append(final_loss)

    # complete trial
    if final_loss is None:
        print("ax trial", trial_index, "failed?")
        ax.log_trial_failure(trial_index=trial_index)
    else:
        ax.complete_trial(trial_index=trial_index,
                          raw_data={'final_loss': (final_loss, 0)})
    print("CURRENT_BEST", ax.get_best_parameters())

    # flush log
    log_msg = "\t".join(map(str, log_record))
    print(log_msg, file=log)
    print(log_msg)
    log.flush()

    # save ax state
    ax.save_to_json_file()
