import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import cv2
from PIL import Image
from pathlib import Path

## Z: unthresholded activation value, W1,2...network length: weights in each layer.
## b: bias used in linear forward, A: thresholded activation values
## Y: Expected output.


## For an autoencoder, use 3 layers with input and output same size.

network_arch = [
    {"layer-size": 7500, "activation": "none"},
    {"layer-size": 3750, "activation": "sigmoid"},
    {"layer-size": 5, "activation": "sigmoid"},
]

def initialize(network_arch, seed = 1):
    np.random.seed(seed)
    param = {}
    num_layers = len(network_arch)
    for i in range(1, num_layers):      # Random weights should be +- n^(-0.5) where n is number of hidden nodes
        param["W" + str(i)] = (np.random.random_sample((network_arch[i]["layer-size"], network_arch[i - 1]["layer-size"])) * 0.03) - 0.015
        param["b" + str(i)] = np.zeros((network_arch[i]["layer-size"], 1))
    return param

## Choice of different activation functions:
def relu(Z):
    R = np.maximum(0, Z)
    return R
def derivative_relu(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
def sigmoid(Z):
    S = 1/(1 + np.exp(-Z))
    return S
def derivative_sigmoid(dA, Z):
    s = sigmoid(Z)
    return dA * (s * (1 - s))
def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
def derivative_tanh(dA, Z):
    tanhh = tanh(Z);
    return dA * (1 - (tanhh * tanhh))

## Iterates forward and returns the Last Activation (AL) and the Forward dictionary
#  that contains the non-thresholded activations (Z) and thresholded activation (A)
def forward_prop(X, param, network_arch):
    forward = {}
    A = X
    num_layers = len(network_arch)
    for i in range(1, num_layers):
        A_prev = A
        W = param["W" + str(i)]
        b = param["b" + str(i)]
        activation = network_arch[i]["activation"]
        Z, A = forward_prop_helper(A_prev, W, b, activation)
        forward["Z" + str(i)] = Z
        forward["A" + str(i)] = A
    forward["A0"] = X
    AL = A
    return AL, forward

## Takes A-previous, the current weights + bias, and function. Outputs both the
# thresholded and non-thresholded activations.
def forward_prop_helper(A_prev, W, b, activation):
    if activation == "relu":
        Z = linear_forward(A_prev, W, b)
        A = relu(Z)
    elif activation  == "sigmoid":
        Z = linear_forward(A_prev, W, b)
        A = sigmoid(Z)
    elif activation == "tanh":
        Z = linear_forward(A_prev, W, b)
        A = tanh(Z)
    return Z, A

## Calculates the activation by finding the dot product between the Weight + bias and
# the previous Activation layer.
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    return Z

## Computes the Error of AL and Expected (Y) using Log means.
def compute_error(AL, Y):
    m = Y.shape[1]
    inside_cost = np.multiply(np.log(AL), Y) + np.multiply(1 - Y, np.log(1 - AL))
    cost = - np.sum(inside_cost) / m
    cost = np.squeeze(cost)      # Gets rid of extra 1-dimensional entries formed from the cost function.
    return cost                 ## Log Loss function

    # omega = Y - AL
    # sum_omega = np.sum(omega * omega)
    # cost = 1.0 / len(Y) * sum_omega
    # return cost                        ## Mean Squared Loss Function

## Runs back prop on the network. dA represents change in activations and is equivalent
# to the "psis" in my Java neural net code. Psis = (Y - AL) * Z(last - 1)
def back_prop(AL, Y, param, forward, network_arch):
    gradients = {}
    num_layers = len(network_arch)
    m = AL.shape[1]         # How many nodes in output
    Y = Y.reshape(AL.shape)     # added this in case there are extra empty arrays or different sized values passed in.
    omegasL = Y - AL
    dAL = -derivative_sigmoid(omegasL, forward["Z" + str(num_layers - 1)])      # Change in predicted output ie. error for output layer.
    dA_prev = dAL

    # print(forward)

    for l in reversed(range(1, num_layers)):
        dA_current = dA_prev

        activation = network_arch[l]["activation"]
        W_current = param["W" + str(l)]
        Z_current = forward["Z" + str(l)]
        A_prev = forward["A" + str(l - 1)]

        dA_prev, dW_current, db_current = back_prop_helper(dA_current, Z_current, A_prev, W_current, activation)

        # print("Delta Weight" + str(l) + ": ")
        # print(dW_current)

        gradients["dW" + str(l)] = dW_current
        gradients["db" + str(l)] = db_current

    return gradients

## Calculates the derivative of the activation function for gradient descent.
def back_prop_helper(dA, Z, A_prev, W, activation):
    if activation == "relu":
        dZ = derivative_relu(dA, Z)         # dA * derivative is equivalent to "psis" which is derivative * omegas (the layer)
        dA_prev, dW, db = linear_backward(dZ, A_prev, W)
    elif activation == "sigmoid":
        dZ = derivative_sigmoid(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, A_prev, W)
    elif activation == "tanh":
        dZ = derivative_tanh(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, A_prev, W)
    return dA_prev, dW, db

## Generalized version of finding deltaWeights, deltaBiases, and helps set A_prev for the next iteration.
def linear_backward(dZ, A_prev, W):
    m = A_prev.shape[1]

    # print("        dZ: ")
    # print(dZ)
    # print("        A_prevT")
    # print(A_prev.T)

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W.T, dZ)           # Finds "layer_2" which is weights * psis

    return dA_prev, dW, db

## Adds the delta weights to the weights.
def add_delta_weights(param, gradients, learning_rate, network_arch):
    num_layers = len(param) // 2
    for i in range(1, num_layers):
        param["W" + str(i)] = param["W" + str(i)] - learning_rate * gradients["dW" + str(i)]
        param["b" + str(i)] = param["b" + str(i)] - learning_rate * gradients["db" + str(i)]
    return param

## TODO: fully implement a batch size option to optimize training.
def batch_add_delta_weights(param, delta_weights, learning_rate, network_arch, batch_size):
    num_layers = len(param) // 2
    for i in range(1, num_layers):
        param["W" + str(i)] = param["W" + str(i)] - learning_rate * 1/batch_size * delta_weights["dW" + str(i)]
        param["b" + str(i)] = param["b" + str(i)] - learning_rate * 1/batch_size * delta_weights["db" + str(i)]
    return param

## Trains the network.
def train_network(X, Y, network_arch, learning_rate = 15.0, max_iterations = 10000, error_threshold = 0.01, print_cost = True):
    np.random.seed(3)
    costs = []
    length = len(network_arch)
    param = initialize(network_arch)
    iteration = 0
    finish = False
    num_training_sets = len(X)
    weights = {}

    while finish == False:
        iteration += 1
        err = 0
        for k in range(1, length):
            weights["dW" + str(k)] = 0
            weights["db" + str(k)] = 0

        # for i in range(num_training_sets):    ## TODO: Split X into batches and change this loop to iterate over each batch

        ## Foward propagation
        AL, forward = forward_prop(X, param, network_arch)

        ## Find error.
        err += compute_error(AL, Y)

        ## Back propagation
        gradients = back_prop(AL, Y, param, forward, network_arch)

        for j in range(1, length):
            weights["dW" + str(j)] += gradients["dW" + str(j)]
            weights["db" + str(j)] += gradients["db" + str(j)]


        ## Update parameters. TODO: When implementing BATCH_SIZE, make sure to update weights outside of the batch gradient descent.
        param = add_delta_weights(param, weights, learning_rate, network_arch)

        costs.append(err)
        print(err)

        if iteration >= max_iterations:
            finish = True
            print("REACHED MAXIMUM ITERATIONS.")
        elif err < error_threshold:
            finish = True
            print("ERROR THRESHOLD REACHED")

    ## Plot the cost for debugging.
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate: " + str(learning_rate))
    plt.show()

    return param

## Prints weights.
def printWeights(param, network_arch):
    num_layers = len(network_arch)
    for i in range(1, num_layers):
        print(param["W" + str(i)])

## Prints outputs. NOT CURRENTLY USED.
def printOutputs(forward, network_arch):
    num_layers = len(network_arch)
    print(forward["A" + num_layers])

## Predicts output by rounding final outputs.
def predict(AL):
    prediction = np.zeros([AL.shape[0], AL.shape[1]])
    for r in range(AL.shape[0]):
        for c in range(AL.shape[1]):
            if AL[r,c] < 0.5:
                prediction[r, c] = 0
            else:
                prediction[r, c] = 1
    return prediction

## Sets weights to the network from given files.
def set_weights(weightfile, biasfile, network_arch):
    param = {}
    num_layers = len(network_arch)
    for i in range(1, num_layers):
        param["W" + str(i)] = np.zeros((network_arch[i]["layer-size"], network_arch[i - 1]["layer-size"]))
        param["b" + str(i)] = np.zeros((network_arch[i]["layer-size"], 1))
    wf = Path(weightfile)
    bf = Path(biasfile)
    with wf.open("rb") as f:
         param["W"] = np.load(f)
    with bf.open("rb") as d:
        param["b"] = np.load(d)
    return param

#### Tests autoencoder.
# X = np.array([[[0],[1],[0],[0],[1]]])
# Y = np.array([[[1],[1],[1],[1],[1]]])
# parameters = train_network(X, Y, network_arch)
# print("WEIGHTS FROM TRAINED NETWORK: ")
# printWeights(parameters, network_arch)
# print("\n\n")
#
# ## Run a test for autoencoder.
# test = np.array([[0],[0],[1],[0],[1]])
# print("OUTPUTS FROM TRAINED NETWORK: ")
# AL, forward = forward_prop(test, parameters, network_arch)
# print(AL)



#### Tests single sample.
# X = np.array([[[0], [1]]])
# Y = np.array([[[1]]])

## Tests multiple samples XOR with easier formatting.
# x0 = np.array([[0], [0]])
# x1 = np.array([[0], [1]])
# x2 = np.array([[1], [0]])
# x3 = np.array([[1], [0]])
# x4 = np.array([[1], [1]])
# x5 = np.array([[0], [1]])
# x6 = np.array([[0], [0]])
# x7 = np.array([[1], [1]])
# X = np.array([x0,x1,x2,x3,x4,x5,x6,x7])
# y0 = np.array([[0]])
# y1 = np.array([[1]])
# y2 = np.array([[1]])
# y3 = np.array([[1]])
# y4 = np.array([[0]])
# y5 = np.array([[1]])
# y6 = np.array([[0]])
# y7 = np.array([[0]])
# Y = np.array([y0,y1,y2,y3,y4,y5,y6,y7])
#
## Runs Test cases for XOR:
# test = np.array([[0],[1]])
# print("Testing (0, 1): ")
# AL, forward = forward_prop(test, parameters, network_arch)
# print(AL)
#
# test = np.array([[1],[0]])
# print("Testing (1, 0): ")
# AL, forward = forward_prop(test, parameters, network_arch)
# print(AL)
#
# test = np.array([[1],[1]])
# print("Testing (1, 1): ")
# AL, forward = forward_prop(test, parameters, network_arch)
# print(AL)


#### Tests for Dibdump using each pixel as input.
# y0 = np.array([[0], [0], [0], [1], [0]])
# y1 = np.array([[1], [0], [0], [0], [0]])
# y2 = np.array([[0], [0], [0], [1], [0]])
# y3 = np.array([[0], [1], [0], [0], [0]])
# y4 = np.array([[0], [0], [1], [0], [0]])
# y5 = np.array([[1], [0], [0], [0], [0]])
# y6 = np.array([[0], [0], [0], [0], [1]])
# y7 = np.array([[0], [1], [0], [0], [0]])
# y8 = np.array([[0], [0], [0], [0], [1]])
#
# Y = np.array([y0,y1,y2,y3,y4,y5,y6,y7,y8])
# print(y0)
#
# def create_input_from_file(filepath, vector_size):
#     x = np.array([])
#     xList = []
#     with open(filepath) as f:
#         for line in f:
#             xList = line.strip().split(" ")
#
#     for val in xList:
#         v = float(val)
#         x = np.append(x, [v], axis=0)
#
#     x = x.reshape(vector_size,1)
#     return x
#
# x0 = create_input_from_file("finger_dibdump_inputs/dibdump5.txt", 2500)
# x1 = create_input_from_file("finger_dibdump_inputs/dibdump1.txt", 2500)
# x2 = create_input_from_file("finger_dibdump_inputs/dibdump8.txt", 2500)
# x3 = create_input_from_file("finger_dibdump_inputs/dibdump3.txt", 2500)
# x4 = create_input_from_file("finger_dibdump_inputs/dibdump4.txt", 2500)
# x5 = create_input_from_file("finger_dibdump_inputs/dibdump0.txt", 2500)
# x6 = create_input_from_file("finger_dibdump_inputs/dibdump6.txt", 2500)
# x7 = create_input_from_file("finger_dibdump_inputs/dibdump2.txt", 2500)
# x8 = create_input_from_file("finger_dibdump_inputs/dibdump9.txt", 2500)
# X = np.array([x0,x1,x2,x3,x4,x5,x6, x7,x8])

#### Test for Neural net with Finger Files with Images as a node.
def image_folder_to_array(folder, height, width):
    images = []
    count = 0
    for file in os.listdir(folder):
        # print(file)
        img = cv2.imread(os.path.join(folder, file))
        if img is not None:
            images.append(img)
            count += 1
    result = np.zeros([count, height, width, 3], dtype = np.uint8)

    picIndex = 0
    for image in images:
        b, g, r  = image[:, :, 0], image[:, :, 1], image[:, :, 2] # For RGB image
        result[picIndex, :, :, 0] = b
        result[picIndex, :, :, 1] = g
        result[picIndex, :, :, 2] = r
        picIndex += 1
    return result

X_orig = image_folder_to_array("finger_images", 50, 50)
X_color = X_orig.reshape(X_orig.shape[0], -1).T
X = X_color / 255.
# print(X.shape)

Y = np.zeros([5, 20])
Y[:,0] = np.array([0, 0, 0, 0, 1])
Y[:,1] = np.array([0, 0, 1, 0, 0])
Y[:,2] = np.array([0, 0, 0, 0, 1])
Y[:,3] = np.array([0, 0, 0, 0, 1])
Y[:,4] = np.array([0, 0, 1, 0, 0])
Y[:,5] = np.array([1, 0, 0, 0, 0])
Y[:,6] = np.array([1, 0, 0, 0, 0])
Y[:,7] = np.array([0, 1, 0, 0, 0])
Y[:,8] = np.array([0, 0, 0, 1, 0])
Y[:,9] = np.array([0, 0, 1, 0, 0])
Y[:,10] = np.array([1, 0, 0, 0, 0])
Y[:,11] = np.array([0, 0, 1, 0, 0])
Y[:,12] = np.array([0, 1, 0, 0, 0])
Y[:,13] = np.array([0, 0, 0, 0, 1])
Y[:,14] = np.array([0, 1, 0, 0, 0])
Y[:,15] = np.array([0, 1, 0, 0, 0])
Y[:,16] = np.array([1, 0, 0, 0, 0])
Y[:,17] = np.array([0, 0, 0, 1, 0])
Y[:,18] = np.array([0, 0, 0, 1, 0])
Y[:,19] = np.array([0, 0, 0, 1, 0])

parameters = train_network(X, Y, network_arch)
print("WEIGHTS FROM TRAINED NETWORK: ")
printWeights(parameters, network_arch)
layers = len(network_arch)
allW = Path("trained_weights.txt")
allB = Path("trained_bias.txt")
with allW.open('ab') as f:
    for w in range(1, layers):
        np.save(f, parameters["W" + str(w)])
with allB.open('ab') as d:
    for b in range(1, layers):
        np.save(d, parameters["b" + str(b)])
for i in range(1, layers):
    wf = Path("weights/w" + str(i) + ".txt")
    bf = Path("bias/b" + str(i) + ".txt")
    weights = parameters["W" + str(i)]
    bias = parameters["b" + str(i)]
    with wf.open('ab') as f:
        np.save(f, weights)
    with bf.open('ab') as s:
        np.save(s, bias)
print("Saved Weights!\n\n")

## Run Test for Finger Files with Test Cases with each image a node:
print("Testing From Trained Network: ")

AL, forward = forward_prop(X, parameters, network_arch)
print("Actual Outputs: ")
print(AL.T)
print("Predicted Outputs Based on Actual Outputs: ")
prediction = predict(AL)
print(prediction)
print("Expected Outputs: ")
print(Y)

param = set_weights("trained_weights.txt", "trained_bias.txt", network_arch)

## TODO: Implement a test case option where the user adds any input to the network
# and the network should run using the saved weights.


# ## Run test cases for Finger Files USES BY PIXEL:
# test = create_input_from_file("dibdump7.txt",2500)
# print("OUTPUTS FROM TRAINED NETWORK: ")
# print("Testing 3 fingers: ")
# AL, forward = forward_prop(test, parameters, network_arch)
# print(AL)
#
# test2 = x0
# print("Testing 1 finger: ")
# AL, forward = forward_prop(test2, parameters, network_arch)
# print(AL)
