import numpy as np
from sympy import C
import matplotlib.pyplot as plt
import matplotlib.colors as c
import pandas as pd



# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).


class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.W = np.ones((3,2), dtype=float)


    def softmax(self):
        v = np.dot(self.X, self.W)
        return (np.exp(v).T/np.exp(v).sum(axis=1)).T
    
    def one_hot(self):
        for i in range(np.zeros((self.y.shape[0],3)).shape[0]):
            np.zeros((self.y.shape[0],3))[i,self.y[i]] = 1
        self.t = np.zeros((self.y.shape[0],3))
        return

    def grad(self):
        for i in range(200000):
            gradient = np.dot((self.softmax() - self.t), self.X.T) + 2*self.lam * np.vstack((np.zeros(3), self.W.T[1:])).T
            self.W = self.W - self.eta * gradient
        return

    # TODO: Implement this method!
    def fit(self, X, y):
        self.X = X
        self.X = np.c_[np.ones(self.X.shape[0]), self.X]
        self.y = y
        self.one_hot()
        self.grad()
        return
        

    # TODO: Implement this method!
    def predict(self, X_pred):
        self.X = X_pred
        pred = self.softmax()
        label = pred.argmax(axis=1)
        return label

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        plt.figure()
        plt.title('Logistic Regression $\mu=$' + str(self.eta) + ' and $\lambda=$' + str(self.lam))
        plt.plot(range(self.runs), self.loss)
        plt.savefig(output_file + '.png')
        if show_charts:
            plt.show()

