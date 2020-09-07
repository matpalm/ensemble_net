import data
import models
import jax.numpy as jnp
import objax
from objax.functional.loss import cross_entropy_logits_sparse
from tqdm import tqdm
import util
import wandb
import sys
import util


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

    if opts.input_mode not in ['single', 'multiple']:
        raise Exception("invalid --input-mode value")
    single_input_mode = opts.input_mode == 'single'
    if not single_input_mode and opts.num_models == 1:
        raise Exception("--num-models must be >1 when --input-mode=multiple")

    run = util.DTS()

    # init w & b
    wandb_enabled = opts.group is not None
    if wandb_enabled:
        wandb.init(project='ensemble_net', group=opts.group, name=run,
                   reinit=True)
        wandb.config.input_mode = opts.input_mode
        wandb.config.num_models = opts.num_models
        wandb.config.dense_kernel_size = opts.dense_kernel_size
        wandb.config.seed = opts.seed
        wandb.config.learning_rate = opts.learning_rate
        wandb.config.batch_size = opts.batch_size
    else:
        print("not using wandb", file=sys.stderr)

    # construct model
    if opts.num_models == 1:
        net = models.NonEnsembleNet(num_classes=10,
                                    dense_kernel_size=opts.dense_kernel_size,
                                    seed=opts.seed)
    else:
        net = models.EnsembleNet(num_models=opts.num_models,
                                 num_classes=10,
                                 dense_kernel_size=opts.dense_kernel_size,
                                 seed=opts.seed)
        # in single input mode the ensemble model must produce a single output
        # for training (since the single input only has a single label). in
        # multiple input mode the model can produce multiple outputs that
        # correspond to the multiple labels.
        net.single_result = single_input_mode

    # setup loss fn and optimiser.

    # in single input mode training the net produces the standard (B, C) logits
    # that are compared to (B,) labels. this is regardless of the number of
    # models (since in single inputs mode we reduce the ensemble output
    # to one output). this is also the form of the loss called during validation
    # loss calculation where the imgs, labels is the entire split.
    def cross_entropy(imgs, labels):
        logits = net.logits(imgs, single_result=True)
        return jnp.mean(cross_entropy_logits_sparse(logits, labels))

    # in multiple input mode we get an output per model; so the logits are
    # (num_models, batch_size, num_classes) i.e. (M, B, C) with labels
    # (M, B). in this case we flatten the logits to (M*B, C) and the labels
    # to (M*B,) for the cross entropy calculation.
    def nested_cross_entropy(imgs, labels):
        logits = net.logits(imgs, single_result=False)
        m, b, c = logits.shape
        logits = logits.reshape((m*b, c))
        labels = labels.reshape((m*b,))
        return jnp.mean(cross_entropy_logits_sparse(logits, labels))

    if single_input_mode:
        gradient_loss = objax.GradValues(cross_entropy, net.vars())
    else:
        gradient_loss = objax.GradValues(nested_cross_entropy, net.vars())
    optimiser = objax.optimizer.Adam(net.vars())

    # create jitted training step
    def train_step(imgs, labels):
        grads, _loss = gradient_loss(imgs, labels)
        optimiser(opts.learning_rate, grads)
    train_step = objax.Jit(train_step,
                           gradient_loss.vars() + optimiser.vars())

    # create jitted call for validation loss
    calculate_validation_loss = objax.Jit(cross_entropy, net.vars())

    # read entire validation set
    validation_imgs, validation_labels = data.validation_dataset()

    # run some epoches of training
    early_stopping = util.EarlyStopping()
    for epoches in tqdm(range(opts.epochs)):
        # make one pass through training set
        if single_input_mode:
            num_inputs = 1
        else:
            num_inputs = opts.num_models
        train_ds = data.training_dataset(batch_size=opts.batch_size,
                                         num_inputs=num_inputs)
        for imgs, labels in train_ds:
            train_step(imgs, labels)
        # check validation loss and early stopping
        validation_loss = float(calculate_validation_loss(validation_imgs,
                                                          validation_labels))
        if early_stopping.should_stop(validation_loss):
            break

    # save model
    # TODO: in early stopping case we can save the prior checkpoint with the
    #       best performance
    model_save_file = "saved_models/"
    if opts.group:
        model_save_file += f"{opts.group}/"
    model_save_file += f"{run}/final.npz"
    util.ensure_dir_exists_for_file(model_save_file)
    print("saved final model to", model_save_file)
    objax.io.save_var_collection(model_save_file, net.vars())

    # final validation metrics
    validation_predictions = net.predict(validation_imgs, single_result=True)
    validation_accuracy = util.accuracy(validation_predictions,
                                        validation_labels)

    # close out wandb run
    if wandb_enabled:
        wandb.log({'validation_loss': validation_loss,
                   'validation_accuracy': validation_accuracy},
                  step=epoches)
        wandb.join()
    else:
        print("final validation_loss", validation_loss)
        print("final validation accuracy", validation_accuracy)

    # return validation loss to ax
    return validation_loss


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--group', type=str,
                        help='w&b init group', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-models', type=int, default=1)
    parser.add_argument('--input-mode', type=str, default='single',
                        help="whether inputs are across all models (single) or"
                        " one input per model (multiple). inv")
    parser.add_argument('--dense-kernel-size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=2)
    opts = parser.parse_args()
    print(opts, file=sys.stderr)

    train(opts)
