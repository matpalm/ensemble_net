import data
import model
import jax.numpy as jnp
import objax
from objax.functional.loss import cross_entropy_logits_sparse
from tqdm import tqdm
import util
import wandb
import sys
import datetime


def train(opts):
    # init w & b
    wandb_enabled = opts.group is not None
    if wandb_enabled:
        if opts.run is None:
            run = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            run = opts.run
        wandb.init(project='ensemble_net',
                   group=opts.group, name=run,
                   reinit=True)
        # wandb.config.num_models = opts.num_models
        wandb.config.dense_kernel_size = opts.dense_kernel_size
        wandb.config.seed = opts.seed
        wandb.config.learning_rate = opts.learning_rate
        wandb.config.batch_size = opts.batch_size
    else:
        print("not using wandb", file=sys.stderr)

    # construct model
    net = model.NonEnsembleNet(num_classes=10,
                               dense_kernel_size=opts.dense_kernel_size,
                               seed=opts.seed)

    # setup loss fn and optimiser
    def cross_entropy(imgs, labels):
        logits = net.logits(imgs)
        return jnp.mean(cross_entropy_logits_sparse(logits, labels))

    gradient_loss = objax.GradValues(cross_entropy, net.vars())
    optimiser = objax.optimizer.Adam(net.vars())

    # create jitted training step
    def train_step(imgs, labels):
        grads, _loss = gradient_loss(imgs, labels)
        optimiser(opts.learning_rate, grads)
    train_step = objax.Jit(train_step,
                           gradient_loss.vars() + optimiser.vars())

    # reate jitted call for validation loss
    calculate_validation_loss = objax.Jit(cross_entropy, net.vars())

    # read entire validation set
    validation_imgs, validation_labels = data.dataset('validate')

    # run some epoches of training
    early_stopping = util.EarlyStopping()
    for epoches in tqdm(range(opts.epochs)):
        # make one pass through training set
        train_ds = data.dataset('train', batch_size=opts.batch_size)
        for imgs, labels in train_ds:
            train_step(imgs, labels)
        # check validation loss and early stopping
        validation_loss = float(calculate_validation_loss(validation_imgs,
                                                          validation_labels))
        if early_stopping.should_stop(validation_loss):
            break

    # close out wandb run
    if wandb_enabled:
        wandb.log({'validation_loss': validation_loss}, step=epoches)
        wandb.join()

    # return validation loss to ax
    return validation_loss


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--group', type=str,
                        help='w&b init group', default=None)
    parser.add_argument('--run', type=str,
                        help='w&b init run', default=None)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--num-models', type=int, default=20)
    parser.add_argument('--dense-kernel-size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=2)
    opts = parser.parse_args()
    print(opts, file=sys.stderr)

    train(opts)
