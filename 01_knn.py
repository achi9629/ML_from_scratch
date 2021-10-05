# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:38:20 2021

@author: HP
"""
from sklearn import datasets
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
    

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
    
class KNN:
    
    def __init__(self,k):
        self.k = k
        
    def fit(self,X,Y):
        self.X_train = X
        self.Y_train = Y
    
    def predict(self, X):     
        y_hat = [self._predict(x) for x in X]
        return y_hat
    
    def _predict(self, x):
        
        # Compute distances between x and all examples in the training set
        distances = [ euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels  = self.Y_train[indices]
        
        # return the most common class label
        most_common_index = Counter(k_neighbor_labels).most_common(1)
        
        return most_common_index[0][0]
        

if __name__ == "__main__":
        iris = datasets.load_iris()
        data, lables = iris.data, iris.target
        
        X_train, X_test, Y_train, Y_test = train_test_split(data, lables, 
                                                            test_size = 0.2,
                                                            shuffle=True,
                                                            random_state = 1234)
        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        
        
        k = 5
        #From scratch
        obj = KNN(k)
        obj.fit(X_train, Y_train)
        y_hat1 = obj.predict(X_test)
        print(np.sum(y_hat1 == Y_test)/len(Y_test)*100)
        
        color_train = ['blue' if y ==0 else 'green' if y==1 else 'red' for y in Y_train]
        color_test = ['blue' if y ==0 else 'green' if y==1 else 'red' for y in y_hat1]
        
        plt.scatter(X_train[:,2], X_train[:,3], marker = 'o', c = color_train)
        plt.scatter(X_test[:,2], X_test[:,3], marker = 's', c = color_test)
        plt.show()
        
        #Using inbuilt functions
        obj = KNeighborsClassifier(k)
        obj.fit(X_train, Y_train)
        y_hat2 = obj.predict(X_test)
        print(np.sum(y_hat2 == Y_test)/len(Y_test)*100)
        
        
