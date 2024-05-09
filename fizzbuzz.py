"""
FizzBuzz is the following:

For each number 1-100:
- if n is divisible by 3, print "fizz"
- if n is divisible by 5, print "buzz"
- if n is divisible by both 5 and 3, print "fizzbuzz"
- otherwise, print the number
"""
from typing import List
import numpy as np

from ryonn.train import train
from ryonn.nn import NeuralNetwork
from ryonn.layers import Linear, Tanh
from ryonn.optim import SGD


def encode_fizzbuzz(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> List[int]:
    """
    10 digit binary encoding of x
    """
    return [x >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encode(x)
    for x in range(101, 1024)
])

targets = np.array([
    encode_fizzbuzz(x)
    for x in range(101, 1024)
])


net = NeuralNetwork([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])


train(net,
      inputs,
      targets,
      num_epochs=5000,
      optimizer=SGD(lr=0.001))

for x in range(1, 101):
    predicted = net.forward_pass(binary_encode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(encode_fizzbuzz(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])


