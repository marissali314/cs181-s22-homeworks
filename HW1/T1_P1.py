#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np
import math
import matplotlib.pyplot as plt

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

print(data[0][0])
print(data[1][1])

def compute_loss(tau):
    # TODO
    
    loss = 0
    for i in range(len(data)):
        sum = 0
        for j in range(len(data)):
            if data[i][0] != data[j][0]:
                sum += math.pow(math.e,((-(data[j][0] - data[i][0])**2 ) / tau ))* (data[j][1])
        loss += math.pow(data[i][1] - sum, 2)
    return loss

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))


# plot (x^*, f(x^*)) for each of the length scales tau above, 0.1 width from x^* = 0 to 12

xrange = np.arange(0, 12, 0.1)
y_pred = []

for tau in (0.01, 2, 100):
    for x in xrange:
        y_sum = 0
        for i in range(len(data)):
            y_sum += math.pow(math.e, -(data[i][0]-x)**2 / tau)*data[i][1]
        y_pred.append(y_sum)

    # naming the x axis
    plt.xlabel('x^*')
    # naming the y axis
    plt.ylabel('f(x^*)')
    
    # giving a title to my graph
    plt.title('Predicted Data with Tau = ' + str(tau))

    plt.plot(xrange, y_pred)
    plt.show()


