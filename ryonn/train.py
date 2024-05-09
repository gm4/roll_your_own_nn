"""
This function trains the NN.
"""

from ryonn.tensor import Tensor
from ryonn.nn import NeuralNetwork
from ryonn.loss import Loss, MSE
from ryonn.optim import Optimizer, SGD
from ryonn.data import DataIterator, BatchIterator

def train(net: NeuralNetwork,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None: 

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward_pass(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward_pass(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)