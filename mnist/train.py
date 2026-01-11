"""
mlp: input -> linear -> relu -> linear -> ce -> loss
"""

import numpy as np

from layers import linear as l, relu as r
from loss_functions import ce
from mnist.mnist_loader.loader import load_mnist

np.random.seed(42)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()

    # hyperparameter: neurons in first layer
    n1 = 100

    x = (X_train.astype(float) / 255.0).reshape(-1, 784)

    # input linear layer
    w1 = np.random.randn(784, n1) * np.sqrt(2 / 784)  # kaiming init for stability
    b1 = np.random.randn(n1)

    # output linear layer
    w2 = np.random.randn(n1, 10) * np.sqrt(2 / n1)  # kaiming init for stability
    b2 = np.random.randn(10)

    def forward(step: int):
        # non-linear activation
        l1_ = l.Linear.np.forward(x, w1, b1)
        r_ = r.Relu.np.forward(l1_[0])
        l2_ = l.Linear.np.forward(r_[0], w2, b2)

        y_ohe = np.eye(10)[y_train]
        loss = ce.CE.np.forward(l2_[0], y_ohe)

        print("step", step, "loss", loss)

    forward(0)
