import numpy as np

class Neuron:

    def __init__(self, n_id=None, inputs=None, weights=None, bias=None, tr_func=None, tr_func_der=None):
        self.id = n_id
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.transfer_function = tr_func
        self.transfer_function_derivative = tr_func_der

    def __str__(self):
        return "Id : " + str(self.id) + "\n" + "inputs : " + str(self.inputs) + "\n" + "weights : " + str(self.weights) + "\n" + "bias : " + str(self.bias) + "\n"

    def fire_neuron(self):
        result = np.dot(self.weights, self.inputs) + self.bias
        apply_func = [[self.transfer_function(e) for e in u] for u in result]
        return apply_func
