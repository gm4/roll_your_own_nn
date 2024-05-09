"""
A loss function measures how good our model predictions are. 
We use this to assess & adjust the parameters of the neural network.
"""
import numpy as np 

from ryonn.tensor import Tensor

class Loss:
    """
    This is just a base function, so not implemented
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    MSE is mean squared error, although we're just going
    to measure total squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.mean((predicted - actual) ** 2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
