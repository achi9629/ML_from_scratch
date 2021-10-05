# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 07:46:10 2021

@author: HP
"""
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split


class Perceptron:
    
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
        
        while True:
            m = 0
            
            for idx,x_i in enumerate(X):
                if Y[idx]*(np.dot(x_i, self.weights) + self.bias) <= 0:
                    print('yes\n')
                    self.weights += Y[idx]*x_i
                    self.bias += Y[idx]
                    m+=1
            if m==0:
                break
                
        
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        
        y_hat = self._activation(linear_output)
        return y_hat
        
    def _unit_step(self, X):
        return np.where(X >0, 1, -1)
    
    def plot(self, X, Y):
        
        plt.scatter(X[:,0], X[:,1], marker = 'o', c = Y )
    
        X1 = [np.min(X[:,0]), np.max(X[:,0])]
        
        m = -self.weights[0]/self.weights[1]
        c = -self.bias/obj.weights[1]
        
    
        X2 = np.multiply(m,X1) + c
        plt.plot(X1, X2, 'black')
        plt.show()
        
        
        
 
#%%    
# Testing
if __name__ == "__main__":
    X, Y = datasets.make_blobs(n_samples=150, 
                               n_features=2, 
                               centers=2, 
                               cluster_std=1.05, 
                               random_state=1234,
                               shuffle = True)
    
    Y = np.where(Y == 1, 1, -1 )
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size = 0.2, 
                                                        shuffle = True, 
                                                        random_state = 1234)
    
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)    
    
    learning_rate = 0.001
    epochs = 1000
    
    #%%
    #From Scratch
    obj = Perceptron(learning_rate, epochs)
    obj.fit(X, Y)
    Y_hat = obj.predict(X_test)

    print('Perceptron Accuracy" ',np.mean(Y_hat==Y_test)*100)
    
    print(obj.weights)
    print(obj.bias)
    
    obj.plot(X, Y)
    # plt.scatter(X[:,0], X[:,1], marker = 'o', c = Y )
    
    # X1 = [np.min(X[:,0]), np.max(X[:,0])]
    # print(X1)
    
    # m = -obj.weights[0]/obj.weights[1]
    # c = -obj.bias/obj.weights[1]
    

    # X2 = np.multiply(m,X1) + c
    # plt.plot(X1, X2, 'black')
    # plt.show()
