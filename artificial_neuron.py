import math

def sigmoid(x):

    res = 1.0 / (1 + math.exp(-x))

    return res

def activate(inputs, weights):

    h = 0

    for (x, w) in zip(inputs, weights):
        h += x*w

    return sigmoid(h)

if __name__ == '__main__':
    inputs = [0.5, 0.3, 0.3]
    weights = [0.4, 0.7, 0.2]
    output = activate(inputs, weights)
    print(output)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
