"""
A neural network is just a collection of layers. 
"""
from typing import Sequence, Iterator, Tuple

from ryonn.tensor import Tensor
from ryonn.layers import Layer

class NeuralNetwork:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward_pass(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs

    def backward_pass(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
