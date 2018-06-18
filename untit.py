# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:20:30 2018

@author: Sherin
"""
import numpy as np
import matplotlib.pyplot as plt

def relu_backward(dA, activation_cache):
    Z = activation_cache[0] # z value from activation function
    g_prime = np.where(Z > 0, 1.0, 0.0) # return 1 or zero and mutliply with AL-Y
    return dA * g_prime
def lin_backward(dA, activation_cache):
    return dA
def compute_cost(AL, Y, cost_func):
    m = Y.shape[1]
    if cost_func == 'mse': #mean sqaure error
        cost = 0.5 * 1/m * np.sum((Y-AL)**2)
    cost = np.squeeze(cost)  # (e.g. this turns [[17]] into 17).
    return cost
def initialize_parameters_deep(layer_dims):
    """
   random initailize to weight and bais for all layers
    W -- weights matrix:(size of current layer, size of previous layer)
    b -- bias vector(size of the current layer, 1)
    """
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters
def relu(Z):
    A = Z * (Z > 0)
    return A, [Z]
def lin(Z):
    return Z, [Z]
def linear_forward(A, W, b):
    """
    A -- activations from previous layer (size of previous layer,samples number)
    W -- weights matrix:(size of current layer, size of previous layer)
    b -- bias vector(size of the current layer, 1)
    Returns: Z -- the input of the activation function/cache -- (A,W,B)
    """
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache
def linear_activation_forward(A_prev, W, b, activation):
    """
    A_prev --: (size of previous layer, number of examples)
    W -- weights matrix,(size of current layer, size of previous layer)
    b -- bias vector,(size of the current layer, 1)
    activation -- "relu" or "lin"
    A -- new activation
    cache --(a,w,b,z new from activation)
    """
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "lin":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = lin(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache
def L_model_forward(X, layer_activation, parameters):
    """
    X -- data, numpy array of shape (input size, number of examples)
    AL -- last post-activation value
    caches -- list of caches containing:(a,w,b,z in new result of activation function)          
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                      activation=layer_activation[l-1])
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],
                                          activation=layer_activation[L-1])
    caches.append(cache)
    return AL, caches
def linear_backward(dZ, cache):
    """
    dZ -- from backprop of activation layer
    cache -- from  in the current layer
    dA_prev -- Gradient of the cost with respect to the activation, same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1 / m * np.dot(dZ, A_prev.T) 
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    dA -- AL-Y,post-activation gradient for current layer l
    cache --  (linear_cache, activation_cache) we stored for backward propagation
    activation --  "relu" or "lin"
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)      
    elif activation == "lin":
        dZ = lin_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db
def L_layer_model(X, Y, layers_dims, layer_activation, cost_func, learning_rate, num_iterations, report_interval):
    """   Arguments:
    X -- data (number of examples, number of features)
    Y -- shape (1, number of examples)
    layers_dims -- [input size ,hidden layer size, output size]
    """
    costs = []  
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, layer_activation, parameters)  
        cost = compute_cost(AL, Y, cost_func) 
        grads = L_model_backward(AL, Y, layer_activation, cost_func, caches)        
        parameters = update_parameters(parameters, grads, learning_rate)
        if i % report_interval == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)
    return parameters, costs

def L_model_backward(AL, Y, layer_activation, cost_func, caches):
    """
   AL :new predicts, Y :real values : ACTIVATION : RELU,LIN , COST : MSE ,
    """
    grads = {}
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)  #make Y is the same shape as AL
    dAL = AL - Y 
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  layer_activation[L-1])
    for l in reversed(range(L - 1)):
      
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    layer_activation[l])
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads


def update_parameters(parameters, grads, learning_rate):
    """

    parameters -- weights and biAS
    grads -- DA , DW,Db output of L_model_backward

    """
    L = len(parameters) // 2  
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
    
    return parameters
ex_count = 1000

X = np.random.rand(2, ex_count)
#2 nstances each with 1000 feature

Y = np.exp(-(np.sum(X**2, axis=0, keepdims=True)))

report_interval = 100

activation = ['relu', 'lin']

parameters, cost_log = L_layer_model(X, Y, [2, 60, 1], activation, 'mse', 0.5, 1000, report_interval)

YH, caches = L_model_forward(X, activation, parameters)

print("Mean Error = ", np.mean(np.abs(YH - Y)))

# plot the cost
plt.plot(np.arange(0, report_interval*len(cost_log), report_interval), np.squeeze(cost_log))
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()