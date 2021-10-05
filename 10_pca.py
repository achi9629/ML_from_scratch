# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:19:02 2021

@author: HP
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

class PrincipalComponent:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        
        # Mean centering
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean
        
        # covariance, function needs samples as columns
        covariance = np.cov(X.T)
        
        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        
        eigenvectors = eigenvectors[idxs]
        eigenvalues = eigenvalues[idxs]
        
        # store first n eigenvectors
        self.components = eigenvectors[0 :self.n_components]
        print(self.components.shape)
    
    def transform(self, X):
        
        # project data
        X = X - self.mean       
        return np.dot(X, self.components.T)
    
    
#%%
if __name__ == "__main__":
    
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    
    #From Scratch
    pca = PrincipalComponent(n_components = 2)
    pca.fit(X)
    X_projected  = pca.transform(X)
    
    print(X.shape, X_projected.shape)
    
    plt.scatter(X_projected[:,0], X_projected[:,1], marker = 'o', c = Y)
    plt.colorbar()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
    
    #%%
    #Using Inbuilt Function
    obj = PCA(n_components = 2)
    obj.fit(X)
    X_projected = obj.fit_transform(X)
    
    print(X.shape, X_projected.shape)

    plt.scatter(X_projected[:,0], X_projected[:,1], marker = 'o', c = Y)
    plt.show()