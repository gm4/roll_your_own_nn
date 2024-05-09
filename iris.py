"""
The iris dataset is a traditional example for classification.
This example uses the iris dataset to estimate petal width 
from sepal length, sepal width and petal length. 
The headers for iris.csv are: 
sepal_length,sepal_width,petal_length,petal_width
"""
import numpy as np

from ryonn.train import train
from ryonn.nn import NeuralNetwork
from ryonn.layers import Linear, Tanh
from ryonn.optim import SGD



data = np.genfromtxt("iris.csv", delimiter=",", skip_header=1)

# to randomize training/test 
# data = np.random.shuffle(data)

inputs = data[:100,:-1]
targets = data[:100, 3]

# make targets an 2-dimensional array
targets = np.reshape(targets, (-1, 1))


for i in range(5):
    print(inputs[i]," : ", targets[i])


net = NeuralNetwork([
    Linear(input_size=3, output_size=10),
    Tanh(),
    Linear(input_size=10, output_size=1)
])

train(net, 
      inputs, 
      targets,
      num_epochs=5000,
      optimizer=SGD(lr=0.001))

for x, y in zip(inputs, targets):
    predicted = net.forward_pass(x)
    print(x, predicted, y)