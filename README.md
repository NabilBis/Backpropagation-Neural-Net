# Backpropagation Neural Network

This given neural network illustrate the use of a **backpropagation's algorithm** to approximate the function below :

        g(x)= 1 + sin((π/4)*x);  -2≤x≤2

**NOTE :** the coding of this algorithm follows the explanation provided in this book [Neural Network Design](https://hagan.okstate.edu/NNDesign.pdf) (Chapter 11).

## Neural network's Architecture

It consists of :

- Input layer
- One hidden layer :

  - Neurons : 2.
  - Transfer function (activation function) : **sigmoid function**.

- Output layer:
  - Neurons : 1.
  - Transfer function: **Pureline function**.

![Network's Architecture](https://images.squarespace-cdn.com/content/v1/51d342a0e4b0290bcc56387d/1414538430117-013J6OBJRLNJORGFCEUD/ke17ZwdGBToddI8pDm48kNGhvwK1dHyJEVTMYsDQT29Zw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpyuNHSz5nMm54PQVFA1hE7KfIV1CslcfaIkUKLdPzeHlsm0QOROT0AAQiev9IOwSsI/image-asset.png)

## Code

The algorithm is split into three classes(modules) : NeuralNet, Layer & Neuron.

A neuron is the core of the network. It computes and transmit result to the front layer's neurons.

One or more neurons combined form a layer, respectively two or more layers combined form a neural network.

```
|-- Network
       |-- Layer
           |-- Neuron
```

The main class(entry point to build and run the net) reads from **"net_parameters.json"** to build the network . This file contains the neurons parameters for each layer.

**NOTE :** Initial parameters are randomly set.

```js
"layer_1": {
  "neuron_1": {
    "id": "1_1",
    "weights": [[-0.27]],
    "inputs": [[1]],
    "bias": [[-0.48]],
    "transfer_function": "sigmoid",
    "transfer_function_derivative": "sigmoid_deri"
  },
```
