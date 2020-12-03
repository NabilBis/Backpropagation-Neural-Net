import numpy as np


class Neural_Net:

    def __init__(self, lays):
        self.layers = lays

    def backpropagate(self, inp, target, index=0):

        layer = self.layers[index]
        dic = {}

        if index == len(self.layers) - 1:
            dic['layer_output'] = layer.activate_layer()
            dic['sensivity'] = layer.sensivity(upfront_weights=None,
                                               upfront_sensivity=None,
                                               is_output_layer=True,
                                               target=target
                                               )
            dic['upfront_weights'] = layer.layer_weights
            layer.update_weights(dic['sensivity'], 0.1, np.array(inp))
            layer.update_bias(dic['sensivity'], 0.1)

            return dic

        layer_outputs = layer.activate_layer()

        for neuron in self.layers[index+1].neurons:
            neuron.inputs = layer_outputs

        upfront_result = self.backpropagate(layer_outputs, target, index+1)

        dic['layer_output'] = layer_outputs

        dic['sensivity'] = layer.sensivity(
            upfront_result['upfront_weights'], upfront_result['sensivity'])

        dic['upfront_weights'] = layer.layer_weights

        layer.update_weights(dic['sensivity'], 0.1, np.array(inp))
        layer.update_bias(dic['sensivity'], 0.1)
        return dic

    def train_net(self, training_data):
        i = 1
        for entry in training_data:
            #             print(i)
            for input_layer_neurons in self.layers[0].neurons:
                input_layer_neurons.inputs = entry['input']
            t = self.backpropagate(entry['input'], entry['target'])
            i += 1

    def approximate(self, inp, index=0):
        layer = self.layers[index]

        if index == len(self.layers) - 1:
            result = layer.activate_layer()
            return result

        layer_outputs = layer.activate_layer()

        for neuron in self.layers[index+1].neurons:
            neuron.inputs = layer_outputs

        result = self.approximate(layer_outputs, index+1)
        return result
