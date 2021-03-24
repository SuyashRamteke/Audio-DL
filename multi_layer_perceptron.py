import numpy as np

class MLP:

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

        print("Weights : ", weights)


    def forward_propagate(self, inputs):

        activations = inputs

        for w in self.weights:

            net_inputs = np.dot(activations, w)
            activations = self.sigmoid(net_inputs)

        return activations

    def sigmoid(self, x):
        y = 1.0/(1 + np.exp(x))

        return y

if __name__ == "__main__":

    mlp = MLP()
    inputs = np.random.rand(mlp.num_inputs)
    outputs = mlp.forward_propagate(inputs)

    print("Network Input : {}".format(inputs))
    print("Network Output: {}".format(outputs))















