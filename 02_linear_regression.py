# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 17:07:24 2021

@author: HP
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score

#%%

class LR:
    
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
            
            # approximate y with linear combination of weights and x, plus bias
            predicted = self.predict(X)
            
            # compute gradients
            dw = 1/samples*np.dot(X.T, (predicted - Y))
            db = 1/samples*np.sum(predicted - Y)
            
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated
    
# Testing
if __name__ == "__main__":
    
    X, Y = datasets.make_regression(n_samples=300, 
                                    n_features=1, 
                                    noise=20, 
                                    random_state=1234)
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size = 0.2, 
                                                        shuffle=True, 
                                                        random_state=1234)
    

    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    
#%%
    learning_rate = 0.001
    epochs = 10000
    
    #From scratch
    obj = LR(lr = learning_rate, epochs = epochs)
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_test)
    loss = np.mean((Y_test - Y_hat)**2)
    print(loss)
    
    m1 = plt.scatter(X_train, Y_train, marker = 'o', c = 'c')
    m2 = plt.scatter(X_test, Y_test, marker = 'o', c = 'm')
    plt.plot(X_test, Y_hat,color = 'k')
    plt.colorbar()
    plt.show()
    
    #%%
    #Using inbuilt functions
    obj = LinearRegression()
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_test)
    loss = np.mean((Y_hat - Y_test)**2)
    print(loss)
    
    cmap = plt.get_cmap("viridis")
    m1 = plt.scatter(X_train, Y_train, color=cmap(0.9), s=10,)
    m2 = plt.scatter(X_test, Y_test, color=cmap(0.5), s=10)
    plt.plot(X_test, Y_hat,color = cmap(0))
    plt.colorbar()
    plt.show()
    
    #%%
    alphas=[1e-3, 1e-2, 1e-1, 1]
    obj = Lasso(1)
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_test)
    loss = np.mean((Y_hat - Y_test)**2)
    print(loss)
    
    cmap = plt.get_cmap("viridis")
    m1 = plt.scatter(X_train, Y_train, color=cmap(0.9), s=10,)
    m2 = plt.scatter(X_test, Y_test, color=cmap(0.5), s=10)
    plt.plot(X_test, Y_hat,color = cmap(0))
    plt.colorbar()
    plt.show()
    
    #%%
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    X, Y = data.data, data.target
    print(X.shape)
    alphas = np.logspace(-3, -1, 30)

    scores = [cross_val_score(Ridge(alpha), X, Y, cv=3).mean() for alpha in alphas]
    plt.plot(alphas, scores,c='g')
    scores = [cross_val_score(Lasso(alpha), X, Y, cv=3).mean() for alpha in alphas]
    plt.plot(alphas, scores,c='r')
    plt.show()
                    
        
            