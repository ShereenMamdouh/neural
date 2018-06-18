import numpy as np
from ann import *

ex_count = 1000

X = np.random.rand(2, ex_count)

Y = np.exp(-(np.sum(X**2, axis=0, keepdims=True)))

report_interval = 100

activation = ['relu', 'relu', 'lin']

parameters, cost_log = L_layer_model(X, Y, [2, 50, 10, 1], activation, 'mse', 0.15, 20000, report_interval)

YH, caches = L_model_forward(X, activation, parameters)

print("Mean Error = ", np.mean(np.abs(YH - Y)))

# plot the cost
plt.plot(np.arange(0, report_interval*len(cost_log), report_interval), np.squeeze(cost_log))
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()