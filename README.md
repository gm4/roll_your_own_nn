# Roll Your Own Neural Network

This is a small neural network ("deep learning network") that works for small problems and is intended as a demonstration of a small Python package/library design and some features of deep learning.

RYONN includes modules for defining:

* tensors
* loss functions
* layers
* neural network
* optimization/gradients
* input data
* model training
* and a few example scripts

## Using it

You can create a new problem by copying one of the examples (`xor.py`, `fizzbuzz.py`, `iris.py`) and changing the input data. Both inputs and targets need to be n-dimensional numpy arrays (see `iris.py`).

## To make it your own

Lots of things you can do to practice or make it your own

* tensors
  * build your own tensor class
  * extend it for compatibility with GPUs
* loss functions
  * MSE is but one measure (many others to choose from)
  * add regularization to penalize larger differences
  * cross-entropy measure for multi-class classification
* layers
  * add sigmoid activation layer & other non-linear activation functions
  * add convolutional layers
  * add recurrent layers (LSTM)
* optimizer
  * include a momentum optimization
* training
  * add different accuracy measures
  * blend different models (model averaging)
* examples
  * create a new example

# Attribution

Though not a fork, this repository is inspired from [joelnet](https://github.com/joelgrus/joelnet)by Joel Grus.