# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 21:19:32 2021

@author: HP
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt

class LogReg:
    
    def __init__(self, lr, epochs):
        super(LogReg, self).__init__()
        
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def fit(self, X, Y):
        
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0
        
        for epoch in range(self.epochs):
            
            # approximate y with linear combination of weights and x, plus bias
            predicted = np.dot(X, self.weights) + self.bias
            
            # apply sigmoid function
            y_hat = self.sigmoid(predicted)
            
            # compute gradients
            dw = np.dot(X.T, (y_hat - Y))/samples
            db = np.sum(predicted - Y)/samples
            
            # update parameters
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
            
            
    def predict(self, X):
        mu = np.dot(X, self.weights) + self.bias
        predicted =  self.sigmoid(mu) >= 0.5
        return predicted
        
    def sigmoid(self, mu):
        return 1 / (1 + np.exp(np.maximum(-mu, np.array([-200]*len(mu)))))
    
    
# Testing
if __name__ == "__main__":
    
    #%%
    bc = datasets.load_breast_cancer()
    X, Y = bc.data, bc.target
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size = 0.2, 
                                                        random_state = 1234,           
                                                        shuffle = True)
    
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    
    learning_rate = 0.0001
    epochs = 1000
    
    #From Scratch
    obj = LogReg(learning_rate, epochs)
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_test)   
    print('Logistic Regression Accuracy :',np.sum(Y_hat == Y_test)/len(Y_test))
   
    
    #%%
    #Using inbuilt functions
    obj = LogisticRegression(max_iter = epochs)
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_test)   
    print('Logistic Regression Accuracy :',np.sum(Y_hat == Y_test)/len(Y_test))
    
    
    