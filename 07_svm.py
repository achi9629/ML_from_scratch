# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 15:12:42 2021

@author: HP
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


#%%
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0
        c = 10
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - c*np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
                
                
def get_hyperplane(X, weights, bias, offset):
    m = -weights[0]/weights[1]
    c = bias/weights[1]
    off = -offset/weights[1]
    return np.multiply(X,m) + c + off

#%%
# Testing
if __name__ == "__main__":
    
    X, Y = datasets.make_blobs(n_samples=250, 
                               n_features=2, 
                               centers=2, 
                               cluster_std=1.05, 
                               random_state=2,
                               shuffle = True)
    
    print(X.shape, Y.shape)

    learning_rate = 0.001
    epochs = 1000
    lembda = 0.01

    #From scratch
    obj = SVM(learning_rate,lembda,epochs)
    obj.fit(X, Y)         
    
    print(obj.w, obj.b)
    
    X1_neg = [np.min(X[:,0]), np.max(X[:,0])]
    
    X2_neg = get_hyperplane(X1_neg, obj.w, obj.b, -1)
    X2_med = get_hyperplane(X1_neg, obj.w, obj.b, 0)
    X2_pos = get_hyperplane(X1_neg, obj.w, obj.b, 1)

    
    plt.scatter(X[:,0],X[:,1],marker = 'o', c = Y)
    plt.plot(X1_neg, X2_neg, 'y--')
    plt.plot(X1_neg, X2_med, 'k')
    plt.plot(X1_neg, X2_pos, 'y--')
    plt.show()