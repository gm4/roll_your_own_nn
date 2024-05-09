"""
We use an optimizer to adjust the network parameters
(w, b) based on the gradients computed during backpropogation.
"""
from ryonn.nn import NeuralNetwork

class Optimizer:
    def step(self, net: NeuralNetwork) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNetwork) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
