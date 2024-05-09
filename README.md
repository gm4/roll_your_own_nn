# Roll Your Own Neural Network

George Muller

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

## Optional next steps to make it your own

Lots of things you can do to practice or make it your own

* tensors
  * build your own tensor class
  * extend it to make it compatible with GPUs
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
  * add accuracy measures
  * blend different models (model averaging)

