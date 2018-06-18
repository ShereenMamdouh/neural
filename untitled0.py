# -*- coding: utf-8 -*-
"""
Created on Sat May  5 11:16:59 2018

@author: Sherin
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache[0]
    ex = np.exp(-Z)
    g_prime = ex / np.power(1+ex, 2)
    return dA * g_prime


def relu_backward(dA, activation_cache):
    Z = activation_cache[0]
    g_prime = np.where(Z > 0, 1.0, 0.0)
    return dA * g_prime


def lin_backward(dA, activation_cache):
    return dA









