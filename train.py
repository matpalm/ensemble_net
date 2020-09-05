import data
import model
import jax.numpy as jnp
import objax
from objax.functional.loss import cross_entropy_logits_sparse


def train(opts):

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
    validate_loss = objax.Jit(cross_entropy, net.vars())

    # run some epoches of training
    for e in range(opts.epochs):
        train_ds = data.dataset('train', batch_size=opts.batch_size)
        for imgs, labels in train_ds:
            # TODO: use tqdn
            #print("T", e, step, imgs.shape, labels.shape)
            train_step(imgs, labels)

        imgs, labels = data.dataset('validate')
        #print("V", e, imgs.shape, labels.shape)
        # TODO: jit this call ?
        print(e, "validate loss", len(labels), validate_loss(imgs, labels))


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--group', type=str,
    #                     help='w&b init group', default=None)
    # parser.add_argument('--run', type=str,
    #                     help='w&b init run', default=None)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--num-models', type=int, default=20)
    parser.add_argument('--dense-kernel-size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    # parser.add_argument('--ortho-init', type=str, default='True')
    # parser.add_argument('--logit-temp', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=2)
    opts = parser.parse_args()
    print(opts, file=sys.stderr)

    train(opts)
