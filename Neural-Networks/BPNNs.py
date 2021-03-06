from __future__ import division
import math, random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(np.dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    """takes in a neural network (represented as a list of lists of lists of weights)
    and returns the output from forward-propagating the input"""

    outputs = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]  # add a bias input
        output = [neuron_output(neuron, input_with_bias)  # compute the output
                  for neuron in layer]  # for this layer
        outputs.append(output)  # and remember it

        # the input to the next layer is the output of this one
        input_vector = output

    return outputs

def backpropagate(network, input_vector, target):
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # the output * (1 - output) is from the derivative of sigmoid
    output_deltas = [output * (1 - output) * (output - target[i])
                     for i, output in enumerate(outputs)]

    # adjust weights for output layer (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output

    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                     np.dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # adjust weights for hidden layer (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input



if __name__ == "__main__":

    raw_digits = [
        """11111
          1...1
          1...1
          1...1
          11111""",

        """..1..
           ..1..
           ..1..
           ..1..
           ..1..""",

        """11111
           ....1
           11111
           1....
           11111""",

        """11111
           ....1
           11111
           ....1
           11111""",

        """1...1
           1...1
           11111
           ....1
           ....1""",

        """11111
           1....
           11111
           ....1
           11111""",

        """11111
           1....
           11111
           1...1
           11111""",

        """11111
           ....1
           ....1
           ....1
           ....1""",

        """11111
           1...1
           11111
           1...1
           11111""",

        """11111
           1...1
           11111
           ....1
           11111"""]


    def make_digit(raw_digit):
        return [1 if c == '1' else 0
                for row in raw_digit.split("\n")
                for c in row.strip()]
        """
        # 代码等价于
        ls =[]
        for row in raw_digit.split("\n"):
            for c in row.strip():
                if c== '1':
                    ls.append(1)
                else:
                    ls.append(0)
        return ls
        """
        
    inputs = list(map(make_digit, raw_digits))

    targets = [[1 if i == j else 0 for i in range(10)]
               for j in range(10)]

    random.seed(0)  # to get repeatable results
    input_size = 25  # each input is a vector of length 25
    num_hidden = 15  # we'll have 5 neurons in the hidden layer
    output_size = 10  # we need 10 outputs for each input

    # each hidden neuron has one weight per input, plus a bias weight
    hidden_layer = [[random.random() for i in range(input_size + 1)]
                    for i in range(num_hidden)]

    # each output neuron has one weight per hidden neuron, plus a bias weight
    output_layer = [[random.random() for i in range(num_hidden + 1)]
                    for i in range(output_size)]

    # the network starts out with random weights
    network = [hidden_layer, output_layer]

    # 10,000 iterations seems enough to converge
    for i in range(5000):#迭代次数
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

    def predict(input):
        return feed_forward(network, input)[-1]


    for i, input in enumerate(inputs):
        outputs = predict(input)
        print(i, [round(p, 2) for p in outputs])

    print("""
.@@@.
...@@
..@@.
...@@
.@@@.""")
    predicted_result0 = predict([0, 1, 1, 1, 0,   # .@@@.
                                0, 0, 0, 1, 1,   # ...@@
                                0, 0, 1, 1, 0,   # ..@@.
                                0, 0, 0, 1, 1,   # ...@@
                                0, 1, 1, 1, 0])  # .@@@.
    print([round(x, 2) for x in predicted_result0],
          "result is:%d" % predicted_result0.index(max(predicted_result0)))

    print("""
.@@@.
@..@@
.@@@.
@..@@
.@@@.""")
    predicted_result1 = predict([0, 1, 1, 1, 0,   # .@@@.
                                1, 0, 0, 1, 1,   # @..@@
                                0, 1, 1, 1, 0,   # .@@@.
                                1, 0, 0, 1, 1,   # @..@@
                                0, 1, 1, 1, 0])  # .@@@.
    print([round(x, 2) for x in predicted_result1],
          "result is:%d" % predicted_result1.index(max(predicted_result1)))
    
    predicted_result = predict([1,1,1,1,1,
                                1,0,0,0,1,
                                1,0,0,0,1,
                                1,0,0,0,1,
                                1,1,1,1,1])
    print([round(x, 2) for x in predicted_result],
          "result is:%d" % predicted_result.index(max(predicted_result)))


