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
from jax import vmap, value_and_grad, pmap, jit
import models
import data
import pickle
import wandb
import logging


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(
        labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def train(opts):

    run = u.DTS()
    logging.info("starting run %s", run)

    # # init w & b
    wandb_enabled = opts.group is not None
    if wandb_enabled and u.primary_host():
        wandb.init(project='ensemble_net', group=opts.group, name=run,
                   reinit=True)
        # save group again explicitly to work around sync bug that drops
        # group when 'wandb off'
        wandb.config.group = opts.group
        wandb.config.seed = opts.seed
        wandb.config.max_conv_size = opts.max_conv_size
        wandb.config.dense_kernel_size = opts.dense_kernel_size
        wandb.config.models_per_device = opts.models_per_device
        wandb.config.learning_rate = opts.learning_rate
        wandb.config.batch_size = opts.batch_size
        wandb.config.steps_per_batch = opts.steps_per_batch
    else:
        logging.info("not using wandb and/or not primary host")

    logging.info("build_model")
    model = models.build_model(opts)

    num_devices = len(jax.local_devices())
    num_models = num_devices * opts.models_per_device

    # we make two rngs; one that is distinct per host and one
    # that will be common across the pod
    host_rng = jax.random.PRNGKey(opts.seed ^ jax.host_id())
    pod_rng = jax.random.PRNGKey(opts.seed * 2)  # o_O

    logging.info("init models")
    keys = jax.random.split(host_rng, num_models)
    logging.debug("model keys %s" % list(keys))
    representative_input = jnp.zeros((1, 64, 64, 3))
    params = vmap(lambda k: model.init(k, representative_input))(keys)

    logging.info("init optimisers")
    opt = optax.adam(opts.learning_rate)
    opt_states = vmap(opt.init)(params)

    def reshape_for_devices_and_shard(p):
        return u.shard(u.reshape_leading_axis(p, (num_devices,
                                                  opts.models_per_device)))

    logging.info("treemap reshape params")
    params = tree_map(reshape_for_devices_and_shard, params)
    opt_states = tree_map(reshape_for_devices_and_shard, opt_states)

    # -----------------------------------
    # prepare loss / training functions

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

    logging.info("compile pmap update")
    p_update = pmap(update,
                    in_axes=(0, 0, 0, 0, 0),
                    axis_name='device')

    # -----------------------------------
    # prepare evaluation functions

    # plumb batch dimension for models_per_device
    all_models_apply = vmap(model.apply, in_axes=(0, None))
    # plumb batch dimension for num_devices
    all_models_apply = vmap(all_models_apply, in_axes=(0, None))

    def ensemble_logits(params, imgs):
        logits = all_models_apply(params, imgs)
        batch_size = logits.shape[-2]  # since last batch may be smaller
        num_classes = 10
        logits = logits.reshape((-1, batch_size, num_classes))  # (M, B, 10)
        ensemble_logits = jnp.sum(logits, axis=0)               # (B, 10)
        return ensemble_logits

    @jit
    def total_ensemble_xent_loss(params, x, y_true):
        y_pred_logits = ensemble_logits(params, x)
        return jnp.sum(softmax_cross_entropy(y_pred_logits, y_true))

    # --------------------------------
    # run training loop

    for epoch in range(opts.epochs):

        # train for one epoch
        logging.info("data.training_dataset: epoch %d", epoch)

        total_training_loss = 0
        training_num_examples = 0

        # split out a new shuffle seed for this epoch common
        # across pod
        pod_rng, shuffle_seed = jax.random.split(pod_rng)

        # create dataset
        train_ds = data.training_dataset(batch_size=opts.batch_size,
                                         shuffle_seed=shuffle_seed[0],
                                         num_inputs=1,
                                         sample_data=opts.sample_data)

        for imgs, labels in train_ds:

            logging.debug("labels %s" % labels)

            # replicate batch across M devices
            # (M, B, H, W, 3)
            imgs = u.replicate(imgs, replicas=num_devices)
            labels = u.replicate(labels, replicas=num_devices)  # (M, B)

            # run across all the 4 rotations
            # for k in range(4):
            #   rotated_imgs = rot90_imgs(imgs, k)

            # run some steps for this set, each with a different set of
            # dropout idxs
            for _ in range(opts.steps_per_batch):
                host_rng, dropout_key = jax.random.split(host_rng)
                logging.debug("dropout_key %s" % dropout_key[0])
                sub_model_idxs = jax.random.randint(dropout_key, minval=0,
                                                    maxval=opts.models_per_device,
                                                    shape=(num_devices,))
                logging.debug("sub_model_idxs %s" % sub_model_idxs)
                params, opt_states, losses = p_update(params, opt_states,
                                                      sub_model_idxs,
                                                      imgs, labels)
                logging.debug("losses %s" % losses)

                total_training_loss += jnp.sum(losses)
                training_num_examples += len(losses)

        mean_training_loss = float(total_training_loss / training_num_examples)
        logging.info("mean training loss %f", mean_training_loss)

        # post epoch stats collection and housekeeping on primary host only
        if u.primary_host():
            # checkpoint model
            ckpt_file = f"saved_models/{run}/ckpt_{epoch:04d}"
            u.ensure_dir_exists_for_file(ckpt_file)
            with open(ckpt_file, "wb") as f:
                pickle.dump(params, f)

            # run validation
            total_validation_loss = 0
            validation_num_examples = 0
            validation_data = data.validation_dataset(
                batch_size=opts.batch_size,
                sample_data=opts.sample_data)
            for imgs, labels in validation_data:
                total_validation_loss += total_ensemble_xent_loss(params, imgs,
                                                                  labels)
                validation_num_examples += len(labels)
            mean_validation_loss = float(
                total_validation_loss / validation_num_examples)
            logging.info("mean validation loss %f", mean_validation_loss)

            if wandb_enabled:
                wandb.log({'training_loss': mean_training_loss}, step=epoch)
                wandb.log({'validation_loss': mean_validation_loss}, step=epoch)

    # close out wandb run
    if u.primary_host():
        if wandb_enabled:
            wandb.log({'final_validation_loss': mean_validation_loss},
                      step=opts.epochs)
            wandb.join()
        else:
            logging.info("finished %s final validation_loss %f" %
                         (run, mean_validation_loss))
        # return validation loss to ax
        return mean_validation_loss
    else:
        return None


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
    parser.add_argument('--group', type=str,
                        help='w&b init group', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-conv-size', type=int, default=256)
    parser.add_argument('--dense-kernel-size', type=int, default=32)
    parser.add_argument('--models-per-device', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--steps-per-batch', type=int, default=4,
                        help='how many steps to run, each with new random'
                             ' dropout, per batch that is loaded')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--log-level', type=str, default='INFO')
    parser.add_argument('--sample-data', action='store_true',
                        help='set for running test with small training data')
    opts = parser.parse_args()
    print(opts, file=sys.stderr)

    # set logging level
    numeric_level = getattr(logging, opts.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % opts.log_level)
    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(numeric_level)  # logging.INFO)

    train(opts)
