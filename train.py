try:
    import jax_pod_setup
except ModuleNotFoundError:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import util as u
import sys
import optax
from tqdm import tqdm
from jax.tree_util import tree_map, tree_multimap
import jax.numpy as jnp
from jax.lax import psum
from jax import vmap, value_and_grad, pmap
import models
import data
import pickle


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(
        labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def train(opts):
    # check --inputs and --num-models config combo. there are only three combos
    # we support.
    #
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

    # if opts.input_mode not in ['single', 'multiple']:
    #     raise Exception("invalid --input-mode value")
    # single_input_mode = opts.input_mode == 'single'
    # if not single_input_mode and opts.num_models == 1:
    #     print("--num-models must be >1 when --input-mode=multiple",
    #           file=sys.stderr)
    #     return None

    run = u.DTS()
    print("starting run", run)

    # # init w & b
    # wandb_enabled = opts.group is not None
    # if wandb_enabled:
    #     wandb.init(project='ensemble_net', group=opts.group, name=run,
    #                reinit=True)
    #     # save group again explicitly to work around sync bug that drops
    #     # group when 'wandb off'
    #     wandb.config.group = opts.group
    #     wandb.config.input_mode = opts.input_mode
    #     wandb.config.num_models = opts.num_models
    #     wandb.config.max_conv_size = opts.max_conv_size
    #     wandb.config.dense_kernel_size = opts.dense_kernel_size
    #     wandb.config.seed = opts.seed
    #     wandb.config.learning_rate = opts.learning_rate
    #     wandb.config.batch_size = opts.batch_size
    #     wandb.config.model_dropout = opts.model_dropout
    # else:
    #     print("not using wandb", file=sys.stderr)

    print(">build_model")
    model = models.build_model(opts)
    print("<build_model")

    num_devices = len(jax.local_devices())
    num_models = num_devices * opts.models_per_device

    rng = jax.random.PRNGKey(opts.seed ^ jax.host_id())

    keys = jax.random.split(rng, num_models)
    representative_input = jnp.zeros((1, 64, 64, 3))
    print(">init models")
    params = vmap(lambda k: model.init(k, representative_input))(keys)

    print(">init opts")
    opt = optax.adam(opts.learning_rate)
    opt_states = vmap(opt.init)(params)

    def reshape_for_devices_and_shard(p):
        return u.shard(u.reshape_leading_axis(p, (num_devices,
                                                  opts.models_per_device)))

    print("treemap reshape")
    params = tree_map(reshape_for_devices_and_shard, params)
    opt_states = tree_map(reshape_for_devices_and_shard, opt_states)

    def mean_ensemble_xent(params, x, y_true):
        logits = model.apply(params, x)
        logits = psum(logits, axis_name='device')
        return jnp.mean(softmax_cross_entropy(logits, y_true))

    def update(params, opt_state, sub_model_idx, x, y_true):
        # select the sub model & corresponding optimiser state to use
        sub_params = tree_map(lambda v: v[sub_model_idx], params)
        sub_opt_state = tree_map(lambda v: v[sub_model_idx], opt_state)
        # calculate loss and gradients; summing logits over all selected models
        losses, grads = value_and_grad(
            mean_ensemble_xent)(sub_params, x, y_true)
        # apply optimiser
        updates, sub_opt_state = opt.update(grads, sub_opt_state)
        sub_params = optax.apply_updates(sub_params, updates)

        # assign updated values back into params and optimiser state
        def update_sub_model(values, update_value):
            return jax.ops.index_update(values, sub_model_idx, update_value)
        params = tree_multimap(update_sub_model, params, sub_params)
        opt_state = tree_multimap(update_sub_model, opt_state, sub_opt_state)
        # return
        return params, opt_state, losses

    print(">pmap update")
    p_update = pmap(update,
                    in_axes=(0, 0, 0, 0, 0),
                    axis_name='device')

    for e in range(opts.epochs):
        # train for one epoch
        print(">data.training_dataset")
        train_ds = data.training_dataset(batch_size=opts.batch_size,
                                         num_inputs=1)
        for b_idx, (imgs, labels) in enumerate(train_ds):
            # replicate batch across M devices
            # (M, B, H, W, 3)
            imgs = u.replicate(imgs)
            labels = u.replicate(labels)  # (M, B)

            # run across all the 4 rotations
            # for k in range(4):
            #   rotated_imgs = rot90_imgs(imgs, k)

            # run some steps for this set, each with a different set of
            # dropout idxs
            for s in range(opts.steps_per_batch):
                rng, dropout_key = jax.random.split(
                    rng)
                sub_model_idxs = jax.random.randint(dropout_key, minval=0,
                                                    maxval=opts.models_per_device,
                                                    shape=(num_devices,))
                params, opt_states, losses = p_update(params, opt_states,
                                                      sub_model_idxs,
                                                      imgs, labels)
                print(e, s, jnp.mean(losses))

            if b_idx > 3:
                break

        # checkpoint model
        ckpt_file = f"saved_models/{run}/ckpt_{e:04d}"
        u.ensure_dir_exists_for_file(ckpt_file)
        with open(ckpt_file, "wb") as f:
            pickle.dump(params, f)

        # run validation
        # TODO)

    # # set up checkpointing; just need more ckpts than early stopping
    # patience
    # ckpt_dir = "saved_models/"
    # if opts.group:
    #     ckpt_dir += f"{opts.group}/"
    # else:
    #     ckpt_dir += "no_group/"
    # ckpt_dir += run
    # ckpt = objax.io.Checkpoint(logdir=ckpt_dir, keep_ckpts=10)

    # # close out wandb run
    # if wandb_enabled:
    #     wandb.config.early_stopped = early_stopping.stopped()
    #     wandb.log({'final_validation_loss': validation_loss},
    #               step=opts.epochs)
    #     wandb.join()
    # else:
    #     print("finished", run,
    #           " early_stopping.stopped()", early_stopping.stopped(),
    #           " final validation_loss", validation_loss)

    # return validation loss to ax
    # return validation_loss


if __name__ == '__main__':

    # import jax.profiler
    # server = jax.profiler.start_server(9999)
    # print("PROFILER STARTED")
    # import time
    # for i in reversed(range(5)):
    #     print(i)
    #     time.sleep(1)

    import argparse
    import sys

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--group', type=str,
    #                     help='w&b init group', default=None)
    parser.add_argument(
        '--seed', type=int, default=0)
    #parser.add_argument('--num-models', type=int, default=1)
    # parser.add_argument('--input-mode', type=str, default='single',
    #                     help="whether inputs are across all models (single) or"
    #                     " one input per model (multiple). inv")
    parser.add_argument('--max-conv-size', type=int, default=64)
    parser.add_argument('--dense-kernel-size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--models-per-device', type=int, default=2)
    parser.add_argument('--steps-per-batch', type=int, default=4,
                        help='how many steps to run, each with new random'
                             ' dropout, per batch that is loaded')
#    parser.add_argument('--model-dropout', action='store_true')
    opts = parser.parse_args()
    print(opts, file=sys.stderr)

    train(opts)
