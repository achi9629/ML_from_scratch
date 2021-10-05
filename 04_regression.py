# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:26:10 2021

@author: HP
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class BaseClass:
    
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def fit(self, X, Y):
        samples, features = X.shape
        
        self.weights = np.zeros(features)
        self.bias = 0
        
        for _ in range(self.epochs):
            
            predicted = self._approximate(X, self.weights, self.bias)
            
            dw = 1/samples*np.dot(X.T, (predicted - Y))
            db = 1/samples*np.sum(predicted - Y)
            
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return self._predict(X, self.weights, self.bias)
    
    def _predict(self, X, weights, bias):
        raise NotImplementedError
        
    def _approximate(self, X, weights, bias):
        raise NotImplementedError
        
class LinearRegression(BaseClass):
    
        def _approximate(self, X, weights, bias):
            return np.dot(X, weights) + bias
        
        def _predict(self, X, weights, bias):
            return np.dot(X, weights) + bias
        
class LogisticRegression(BaseClass):
    
    def _approximate(self, X, weights, bias):
        predicted = np.dot(X, weights) + bias
        return self._sigmoid(predicted)
    
    def _predict(self, X, weights, bias):
        mu = np.dot(X, weights) + bias
        predicted =  self._sigmoid(mu) >= 0.5
        return predicted
        
    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))
        
# Testing
if __name__ == "__main__":
    
    #Linear Regression
    X, Y = datasets.make_regression(n_samples=300, 
                                    n_features=1, 
                                    noise=20, 
                                    random_state=1234)
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size = 0.2, 
                                                        shuffle=True, 
                                                        random_state=1234)
    

    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    
    learning_rate = 0.001
    epochs = 10000
    
    #From scratch
    obj = LinearRegression(lr = learning_rate, epochs = epochs)
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_test)
    loss = np.mean((Y_test - Y_hat)**2)
    print(loss)
    
    #Logistic Regression
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
    obj = LogisticRegression(learning_rate, epochs)
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_test)   
    print('Logistic Regression Accuracy :',np.sum(Y_hat == Y_test)/len(Y_test))
