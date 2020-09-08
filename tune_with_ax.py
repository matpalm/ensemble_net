
import argparse
import train
from ax.service.ax_client import AxClient
import time
import sys
import util as u
import random
import json
import tensorflow as tf

#tf.config.experimental.set_visible_devices([], "GPU")

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--group', type=str, default=None,
                    help='wandb group. if none, no logging')
parser.add_argument('--config-combo', type=str, default=None,
                    help="one of {single_input_output,"
                         " single_input_multiple_outputs,"
                         " multiple_inputs_multiple_outputs")

# parser.add_argument('--prior-run-logs', type=str, default=None,
#                     help='a comma seperated prior runs to prime ax client with')
cmd_line_opts = parser.parse_args()
if cmd_line_opts.config_combo not in ['single_input_output',
                                      'single_input_multiple_outputs',
                                      'multiple_inputs_multiple_outputs']:
    raise Exception("invalid --config-combo")
print(cmd_line_opts, file=sys.stderr)

# note: if running on display GPU you probably want to run set env var
# something like XLA_PYTHON_CLIENT_MEM_FRACTION=.8 to allow jobs tuned too
# large to faily cleanly with OOM

# we tune for 3 major configuration combos; see train.py for more info.

# a) --input-mode=single --num-models=1 ; single set of inputs, single model
#    this is the baseline non ensemble config.
#
# b) --input-mode=single --num-models=M ; single set of inputs, multiple
#    models single set of inputs and labels. ensemble outputs are summed at
#    logits to produce single output.
#
# c) --input-mode=multiple --num-models=M ; multiple inputs, multiple models
#    multiple inputs (with multiple labels) going through multiple models.
#    loss is still averaged over all models though.

ax_params = [
    {
        "name": "dense_kernel_size",
        "type": "range",
        "bounds": [8, 64],
    },
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [1e-5, 1e-2],
        "log_scale": True,
    },
    # {
    #     "name": "batch_size",
    #     "type": "choice",
    #     "values": [32, 64, 128],
    # },
]

if cmd_line_opts.config_combo != 'single_input_output':
    ax_params.append({
        "name": "num_models",
        "type": "choice",
        "values": [2, 4, 8],
    })

ax = AxClient()
ax.create_experiment(
    name="ensemble_net_tuning",
    parameters=ax_params,
    objective_name="final_loss",
    minimize=True,
)

# if cmd_line_opts.prior_run_logs is not None:
#     for log_tsv in cmd_line_opts.prior_run_logs.split(","):
#         u.prime_ax_client_with_prior_run(ax, log_tsv)

u.ensure_dir_exists("logs/%s" % cmd_line_opts.group)
log = open("logs/%s/ax_trials.tsv" % cmd_line_opts.group, "w")
print("trial_index\tparameters\truntime\tfinal_loss", file=log)


while True:
    parameters, trial_index = ax.get_next_trial()
    log_record = [trial_index, json.dumps(parameters)]
    print("starting", log_record)

    class Opts(object):
        pass
    opts = Opts()

    opts.group = cmd_line_opts.group
    opts.seed = random.randint(0, 1e9)

    # TODO: consider just making input_mode a tunable independent of
    #       num_models and just training loop mark the combo as infeasible.
    if cmd_line_opts.config_combo == 'single_input_output':
        opts.input_mode = 'single'
        opts.num_models = 1
    elif cmd_line_opts.config_combo == 'single_input_multiple_outputs':
        opts.input_mode = 'single'
        opts.num_models = parameters['num_models']
    else:  # config_combo multiple_inputs_multiple_outputs
        opts.input_mode = 'multiple'
        opts.num_models = parameters['num_models']

    opts.dense_kernel_size = parameters['dense_kernel_size']
    opts.batch_size = 32  # parameters['batch_size']
    opts.learning_rate = parameters['learning_rate']
    opts.epochs = 50  # max to run, we also use early stopping

    # run
    start_time = time.time()
    final_loss = train.train_in_subprocess(opts)
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
