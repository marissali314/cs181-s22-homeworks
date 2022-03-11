from tkinter import Y
import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful
import pandas as pd
import math


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        self.num_classes = 3

    # TODO: Implement this method!
    def fit(self, X, y):
        self.X = X
        self.y = y

        mu = np.zeros((2,3))
        prob=[]
        sigma=[]
        values_all=[]

        if self.is_shared_covariance: sigma=np.zeros((2,2))
        for k in range(self.num_classes): 
            values=X[y==k,:]
            values_all.append(values)

            prob.append(float(len(y[y==k]))/float(len(y)))

            mu[0,1]=np.mean([values[:,0]])
            mu[1,k]=np.mean([values[:,1]])
            if not self.is_shared_covariance:
                sigma.append(np.cov(values,rowvar=0))
            if self.is_shared_covariance:
                sigma = sigma+(float(len(values))/float(len(X)))*(np.cov(values, rowvar=0))

        self.mu=mu 
        self.sigma=sigma
        self.prob=prob
    
        return

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        y_hot = np.array(pd.get_dummies(y))

        likelihood = 0
        for k in range(y_hot.shape[1]):
            for i in range(y_hot.shape[0]):
                if self.is_shared_covariance:
                    likelihood += (np.log(2.*math.pi)+(1./2.)*np.log(np.linalg.det(self.sigma))+(1./2.)*np.matmul(np.matmul(np.asmatrix(X[i,:]-self.mu[:,k]),np.linalg.inv(self.sigma)),np.asmatrix(X[i,:]-self.mu[:,k]).T))*y_hot[i,k]-np.log(self.prob[k])
                else:
                    likelihood += (np.log(2. * math.pi) + (1. / 2.) * np.log(np.linalg.det(self.sigma[k])) + (1. / 2.) * np.matmul(
                        np.matmul(np.asmatrix(X[i,:]-self.mu[:,k]), np.linalg.inv(self.sigma[k])), np.asmatrix(X[i,:]-self.mu[:,k]).T)) * y_hot[i, k] - np.log(self.prob[k])
        if self.is_shared_covariance:
            print('shared covariance log likelihood is: '+str(likelihood[0,0])+'\n')
        else:
            print('non-shared covariance log likelihood is: '+str(likelihood[0,0])+'\n')

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        predict=np.zeros((X_to_predict.shape[0],3))
        for k in range(3):
            if not self.is_shared_covariance:
                multi=mvn(self.mu[:,k],self.sigma[k]) 
            if self.is_shared_covariance:
                multi = mvn(self.mu[:, k], self.sigma)
            predict[:,k]=multi.pdf(X_to_predict)*self.prob[k]
        return np.array(np.argmax(predict,axis=1))

