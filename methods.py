import numpy as np


def sigmoid(n):
    return 1/(1+np.exp(-n))


def sigmoid_deri(n):
    return sigmoid(n)*(1-sigmoid(n))


def pureline(n):
    return n


def pureline_deri(n):
    return 1


methods = {'sigmoid': sigmoid,
           'sigmoid_deri': sigmoid_deri,
           'pureline': pureline,
           'pureline_deri': pureline_deri
           }
