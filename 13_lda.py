# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 02:54:37 2021

@author: HP
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LinearDiscriminant:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        
    def fit(self, X, Y):
        
        samples, features = X.shape
        class_lables = np.unique(Y)
        
        # Within class scatter matrix:
        # SW = sum((X_c - mean_X_c)^2 )

        # Between class scatter:
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 
        
        overall_mean = np.mean(X, axis = 0)
        
        S_W = np.zeros((features, features))
        S_B = np.zeros((features, features))
        
        for c in class_lables:
            
            X_c = X[Y == c]          
            mean_c = np.mean(X_c, axis = 0)
            
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            S_W += np.dot((X_c - mean_c).T, X_c - mean_c)
            
            # (4, 1) * (1, 4) = (4,4) -> reshape
            n_c = X_c.shape[0] 
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)       
            S_B += n_c*np.dot(mean_diff, mean_diff.T)
            
            # Determine SW^-1 * SB
            A = np.dot(np.linalg.inv(S_W), S_B)
            
            # Get eigenvalues and eigenvectors of SW^-1 * SB
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            # -> eigenvector v = [:,i] column vector, transpose for easier calculations
            # sort eigenvalues high to low
            eigenvectors = eigenvectors.T
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[idx]
            eigenvalues = eigenvalues[idx]
            
            # store first n eigenvectors
            self.components = eigenvectors[0:self.n_components]
            
            
    def transform(self, X):
        
        return np.dot(X, self.components.T)
            
#%%
if __name__=="__main__": 
           
    iris = load_iris()
    X, Y = iris.data, iris.target
    
    print(X.shape, Y.shape)
    
    #%%
    #From Scratch
    obj = LinearDiscriminant(n_components = 2)
    obj.fit(X, Y)
    
    X_transform = obj.transform(X)
    
    print(X_transform.shape)
    
    plt.scatter(X_transform[:,0], X_transform[:,1], marker = 'o', c = Y)
    
    #%%
    #Using Inbuilt Function
    lda = LinearDiscriminantAnalysis(n_components = 2)
    lda.fit(X, Y)
    X_transform = lda.transform(X)    
    
    print(X_transform.shape)    

    plt.scatter(X_transform[:,0], X_transform[:,1], marker = 'o', c = Y)
