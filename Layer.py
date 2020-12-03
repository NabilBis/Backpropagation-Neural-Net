import numpy as np


class Layer:

    def __init__(self, neur=[]):
        self.neurons = neur
        self.layer_weights = []
        self.layer_bias = []
        self.layer_output = []
        for neuron in self.neurons:
            for row_weights in neuron.weights:
                self.layer_weights.append(row_weights)
            for row_bias in neuron.bias:
                self.layer_bias.append(row_bias)

    def __str__(self):
        return "Layer_weights: " + str(self.layer_weights) + "\n" + "Layer's bias: " + str(self.layer_bias) + "\n"

    def last_layer_sensivity(self, function, layer_output, target):
        jacob = self.__jacobian(function, layer_output)
        error = self.__error(target, layer_output)
        term = np.dot(jacob, error)
        return -2 * term

    def __jacobian(self, function, vals):
        rows = cols = len(vals)
        if cols == 1:
            return [function(vals[0])]
        jab = np.zeros((rows, cols))
        for i in range(rows):
            jab[i][i] = function(vals[i][0])
        return jab

    def __error(self, v_1, v_2):
        return np.subtract(v_1, v_2)

    def sensivity(self, upfront_weights, upfront_sensivity, is_output_layer=False, target=None):
        if is_output_layer:
            return self.last_layer_sensivity(self.neurons[0].transfer_function_derivative,
                                             self.layer_output,
                                             target
                                             )
        first_term = self.__jacobian(self.neurons[0].transfer_function_derivative,
                                     self.layer_output
                                     )
        second_term = np.array(upfront_weights).T
        ad = np.dot(first_term, second_term)

        return np.dot(ad, [upfront_sensivity])

    def update_weights(self, sensivity, rate_of_convergence, layer_input):
        term = np.dot(sensivity, layer_input.T)
        r_time = np.dot(rate_of_convergence, term)
        self.layer_weights = np.subtract(self.layer_weights, r_time)
        for i in range(len(self.neurons)):
            self.neurons[i].weights = [self.layer_weights[i].tolist()]

    def update_bias(self, sensivity, rate_of_convergence):
        first_term = np.dot(rate_of_convergence, sensivity)
        self.layer_bias = np.subtract(self.layer_bias, first_term)
        for i in range(len(self.neurons)):
            self.neurons[i].bias = [self.layer_bias[i].tolist()]

    def activate_layer(self):
        self.layer_output = []
        for neuron in self.neurons:
            for row_in_matrix_result in neuron.fire_neuron():
                self.layer_output.append(row_in_matrix_result)
        return self.layer_output
