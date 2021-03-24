#Implementing gradient descent and backpropagation

import numpy as np
from random import random

class MLP(object):

    def __init__(self, num_inputs=3, num_hidden=[3, 4], num_outputs=3):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [num_inputs] + num_hidden + [num_outputs]

        print("layers", layers)

        #initiate random weights

        weights = []
        for i in range(len(layers) - 1):

            # creating weight matrix
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

        print("Weights : ", weights)


    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0] = activations

        for i, w in enumerate(self.weights):

            net_inputs = np.dot(activations, w)
            activations = self.sigmoid(net_inputs)
            self.activations[i+1] = activations

        return activations

    def back_propagate(self, error, verbose = False):

        for i in reversed(range(len(self.derivatives))):

            activations = self.activations[i+1]
            delta = error * self.sigmoid_derivative(activations)

            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

        if verbose :
            print("Derivatives for W{}: {}:".format(i, self.derivatives[i]))

        return error

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)



    def sigmoid(self, x):
        y = 1.0/(1 + np.exp(-x))

        return y

if __name__ == "__main__":

    mlp = MLP(2, [5], 1)
    #inputs = np.random.rand(mlp.num_inputs)
    #outputs = mlp.forward_propagate(inputs)

    inputs = np.array([0.2, 0.2])
    target = ([0.4])

    output = mlp.forward_propagate(inputs)
    error = target - output

    mlp.back_propagate(error)

    print("Network Input : {}".format(inputs))
    print("Network Output: {}".format(output))















