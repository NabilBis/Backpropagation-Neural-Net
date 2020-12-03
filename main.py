import numpy as np
import json
from Neuron import Neuron
from Layer import Layer
from methods import methods
from NeuralNet import Neural_Net


def set_net_layers(data):
    net_layers = []
    for layers in range(len(data)):
        try:
            layer_params = data['layer_{}'.format(layers+1)]
            layer_neuros = []
            for neuron in range(len(layer_params)):
                neuron_param = layer_params['neuron_{}'.format(neuron+1)]
                neur = Neuron(n_id=neuron_param['id'],
                              inputs=neuron_param['inputs'],
                              weights=neuron_param['weights'],
                              bias=neuron_param['bias'],
                              tr_func=methods[neuron_param['transfer_function']],
                              tr_func_der=methods[neuron_param['transfer_function_derivative']])

                layer_neuros.append(neur)
            layer = Layer(layer_neuros)
            net_layers.append(layer)

        except:
            print("Invalid Json format")
    return net_layers


def training_data():
    def my_function(x):
        return 1+np.sin((np.pi/4)*x)

    array = np.linspace(-2, 2, 50, endpoint=True)

    data_train = []

    for i in array:
        dic = {}
        dic['input'] = [[i]]
        dic['target'] = my_function(i)
        data_train.append(dic)

    r = {
        'input': [[1]],
        'target': 1.707
    }
    data_train.insert(0, r)
    return data_train


if __name__ == "__main__":
    net_paramters = open('net_parameters.json', "r")
    data = json.loads(net_paramters.read())
    net_layers = set_net_layers(data)

    net = Neural_Net(net_layers)
    approximation = net.approximate([[1.22]])
    print(approximation)
    net.train_net(training_data())
    approximation = net.approximate([[1.22]])
    print(approximation)
