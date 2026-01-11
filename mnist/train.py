"""
mlp: input -> linear -> relu -> linear -> ce -> loss
"""

import time

import numpy as np

from layers import linear as l, relu as r
from loss_functions import ce
from mnist.mnist_loader.loader import load_mnist

np.random.seed(42)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()

    # hyperparameter: neurons in first layer
    n1 = 100

    _x = (X_train.astype(float) / 255.0).reshape(-1, 784)

    # input linear layer
    w1 = np.random.randn(784, n1) * np.sqrt(2 / 784)  # kaiming init for stability
    b1 = np.random.randn(n1)

    # output linear layer
    w2 = np.random.randn(n1, 10) * np.sqrt(2 / n1)  # kaiming init for stability
    b2 = np.random.randn(10)

    def forward(x):
        # non-linear activation
        l1_out, l1_cache = l.Linear.np.forward(x, w1, b1)
        r_out, r_cache = r.Relu.np.forward(l1_out)
        l2_out, l2_cache = l.Linear.np.forward(r_out, w2, b2)

        y_ohe = np.eye(10)[y_train]
        loss = ce.CE.np.forward(l2_out, y_ohe)

        return loss, (l1_cache, r_cache, l2_cache, l2_out, y_ohe)

    def backward(l1_cache, r_cache, l2_cache, l2_out, y_ohe):
        back_ce = ce.CE.np.backward(l2_out, y_ohe)
        back_l2 = l.Linear.np.backward(back_ce, l2_cache)
        back_r = r.Relu.np.backward(back_l2.dX, r_cache)
        back_l1 = l.Linear.np.backward(back_r.dX, l1_cache)

        # only l1 & l2 layers have fields that can be modified by gradients.
        return back_l1, back_l2

    def predict(x):
        """Forward pass without loss - just get logits"""
        l1_out, _ = l.Linear.np.forward(x, w1, b1)
        r_out, _ = r.Relu.np.forward(l1_out)
        l2_out, _ = l.Linear.np.forward(r_out, w2, b2)
        return l2_out

    def accuracy(x, y):
        """Compute accuracy: % of correct predictions"""
        logits = predict(x)
        preds = np.argmax(logits, axis=1)
        return (preds == y).mean() * 100

    # 2 hyperparameters
    lr = 0.01
    epochs = int(1e3)

    for i in range(epochs):
        t1 = time.time()
        loss, (_l1_cache, _r_cache, _l2_cache, _l2_out, _y_ohe) = forward(_x)
        l1_grad, l2_grad = backward(_l1_cache, _r_cache, _l2_cache, _l2_out, _y_ohe)
        t2 = time.time()

        if i % 10 == 0:
            print("time", t2 - t1, "epoch", i, "loss", loss)

        w1 -= lr * l1_grad.dW
        b1 -= lr * l1_grad.db

        w2 -= lr * l2_grad.dW
        b2 -= lr * l2_grad.db

        if i % 50 == 0:
            print(
                "epoch",
                i,
                "train",
                accuracy(_x, y_train),
                "test",
                accuracy((X_test.astype(float) / 255.0).reshape(-1, 784), y_test),
            )

    # Evaluate after training
    print(f"\nTrain accuracy: {accuracy(_x, y_train):.2f}%")
    print(
        f"Test accuracy: {accuracy((X_test.astype(float) / 255.0).reshape(-1, 784), y_test):.2f}%"
    )
