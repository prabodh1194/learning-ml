from dataclasses import dataclass
import numpy as np
import torch

from layers.base import Layer, LayerGradients, test_layer_linear


@dataclass
class LinearCache:
    X: np.ndarray
    W: np.ndarray
    b: np.ndarray


class Linear(Layer):
    class np:
        @staticmethod
        def forward(
            X: np.ndarray, W: np.ndarray, b: np.ndarray
        ) -> tuple[np.ndarray, LinearCache]:
            y = X @ W + b
            return y, LinearCache(X, W, b)

        @staticmethod
        def backward(dout: np.ndarray, cache: LinearCache) -> LayerGradients:
            """
            Y = X @ W + b

            - if there are 3 inputs & one neuron, then that neuron will have 3 weights, such that every input can be weighted.
            - we can have multiple neurons. it can be assumed that one neuron is one feature. the inputs are individually & consurrently weighted by the neurons.
            - there can be multiple iterations of inputs as well.

            X is hence B batches. per batch there are T (time) elements.
            W has C (channels) neurons with T weights.

            Y[B, C] = X[B, T] @ W[T, C] + b[C]

            grad w.r.t Y:
            dX = how does varying the inputs affect Y
            dW = how does varying the weights affect Y
            db = how does varying the bias affect Y

            going back to assuming 1 batch of 3 inputs & 1 weight;
            Y = (w1*x1 + w2*x2 + w3*x3) + b
            adding more neurons & batches doesn't change the math. it just adds scale.

            Now loss is a function of Y.
            dout = dL/dY tells the gradient of Y.

            by chain rule: dL/dW = dL/dY * dY/dW
            similarly,
            dL/dX = dL/dY * dY/dX
            dL/dB = dL/dY * dY/dB

            - thumb of rule for gradient math:
              - addition distributes grad equally to all operands.
              - multiplication distributes grad by opposite's scale.
                - if c = a * b; then a.grad = b * c.grad & b.grad = a * c.grad
            - dout is distributed equally to all the weights of a neuron scaled by the corresponding X.
            - every new batch in input adds a new dout in output. in turn the scaled dout piles up at the weight. this is a row-wise implication.
            - every new neuron in the linear layer, adds a new dout in output. this is a column-wise implication.
            """

            dW = cache.X.T @ dout
            dX = dout @ cache.W.T
            db = dout.sum(axis=0)

            return LayerGradients(dX, dW, db)

    class torch:
        @staticmethod
        def forward(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return X @ W + b


if __name__ == "__main__":
    np.random.seed(42)
    # input size is 3. we are doing a batch of 4
    X = np.random.randn(4, 3)
    # layer of 5 neurons
    W = np.random.randn(3, 5)
    b = np.random.randn(5)

    test_layer_linear(Linear, X, W, b)
