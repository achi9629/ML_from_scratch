# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 10:09:38 2021

@author: HP
"""

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

#%%
class Percep:
    
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self._activation = self._unit_step
        self.weights = None
        self.bias = None
        
    def fit(self, X, Y):
        
        samples, features = X.shape
        
        self.weights = np.zeros(features)
        self.bias = 0
        
        index = np.arange(len(Y))
        for i in range(5): index = np.random.permutation(index) 
        X = X[index]
        Y = Y[index]
        for epoch in range(self.epochs):
            
            for idx, x_i in enumerate(X):
                
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_hat = self._activation(linear_output)
                
                update = self.lr*(Y[idx] - y_hat)
                
                self.weights += update*x_i
                self.bias += update
        
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        
        y_hat = self._activation(linear_output)
        return y_hat
        
    def _unit_step(self, X):
        return X >=0
    
    def plot(self, X, Y):
        
        #Plot all points with 2 different colours
        plt.scatter(X[:,0], X[:,1], marker = 'o', c = Y )
        
        #Finding the perceptron line as y = mx + c = 0 (wx = 0)
        X1 = [np.min(X[:,0]), np.max(X[:,0])]
        
        m = -obj.weights[0]/obj.weights[1]
        c = -obj.bias/obj.weights[1]
        
    
        X2 = np.multiply(m,X1) + c
        plt.plot(X1, X2, 'k')
        plt.show()
 
#%%    
# Testing
if __name__ == "__main__":
    X, Y = datasets.make_blobs(n_samples=250, 
                               n_features=2, 
                               centers=2, 
                               cluster_std=1.05, 
                               random_state=1234,
                               shuffle = True)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size = 0.2, 
                                                        shuffle = True, 
                                                        random_state = 1234)
    
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    
    learning_rate = 0.001
    epochs = 1000
    
    
    #%%
    #From scratch
    obj = Percep(learning_rate, epochs)
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_test)
    
    print('Perceptron Accuracy" ',np.mean(Y_hat==Y_test)*100)
    
    print('Perceptron Weights: ',obj.weights)
    print('Perceptron Bias: ',obj.bias)
    
    obj.plot(X, Y)
    
    # #Plot all points with 2 different colours
    # plt.scatter(X[:,0], X[:,1], marker = 'o', c = Y )
    
    # #Finding the perceptron line as y = mx + c = 0 (wx = 0)
    # X1 = [np.min(X[:,0]), np.max(X[:,0])]
    # print(X1)
    
    # m = -obj.weights[0]/obj.weights[1]
    # c = -obj.bias/obj.weights[1]
    

    # X2 = np.multiply(m,X1) + c
    # plt.plot(X1, X2, 'k')
    # plt.show()

#%%
    
    #Using Inbuilt Function
    perceptron = Perceptron(tol=(1e-3), random_state=1234, max_iter = epochs)
    perceptron.fit(X_train, Y_train)
    perceptron.score(X_test, Y_test)    
    Y_hat = perceptron.predict(X_test)
    print('Accuract: ',np.mean(Y_hat == Y_test)*100)
    
    