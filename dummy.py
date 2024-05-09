"""
A dummy example 
"""
import numpy as np

from ryonn.train import train
from ryonn.nn import NeuralNetwork
from ryonn.layers import Linear, Tanh

inputs = np.array([
    [0,0,0,0],
    [1,0,0,0],
    [0,1,0,0],
    [1,1,0,0]
])

# targets = np.array([
#     [1,0],
#     [0,1],
#     [0,1],
#     [1,0]
# ])

targets = np.array([
    [0],
    [1],
    [1],
    [0]
])
print(targets.shape)
net = NeuralNetwork([
    Linear(input_size=4, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, 
      inputs, 
      targets,
      num_epochs=5)

for x, y in zip(inputs, targets):
    predicted = net.forward_pass(x)
    print(x, predicted, y)