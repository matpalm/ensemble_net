import data
import model
import jax.numpy as jnp
import objax
from objax.functional.loss import cross_entropy_logits_sparse

if __name__ == '__main__':
    ds = data.dataset('test', batch_size=4)
    for imgs, labels in ds:
        break

    net = model.NonEnsembleNet(num_classes=10)
    print(jnp.around(net.logits(imgs), 2))
    print(jnp.around(net.predict(imgs), 2))

    def cross_entropy(imgs, labels):
        logits = net.logits(imgs)
        return jnp.mean(cross_entropy_logits_sparse(logits, labels))

    gradient_loss = objax.GradValues(cross_entropy, net.vars())
    optimiser = objax.optimizer.Adam(net.vars())
    lr = 1e-3

    def train_step(imgs, labels):
        grads, loss = gradient_loss(imgs, labels)
        optimiser(lr, grads)
        return loss
    train_step = objax.Jit(train_step,
                           gradient_loss.vars() + optimiser.vars())

    for _ in range(20):
        print(train_step(imgs, labels))

    print(jnp.around(net.predict(imgs), 2))
    print(jnp.argmax(net.predict(imgs), axis=-1))
    print(labels)
