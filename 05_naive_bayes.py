# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 04:46:54 2021

@author: HP
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class NaiveBayes:
    
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        
        samples, features = X.shape
        
        self._classes = np.unique(Y)
        
        # calculate mean, var, and prior for each class
        self._mean = np.zeros((len(self._classes), features), dtype = np.float64)
        self._var =  np.zeros((len(self._classes), features), dtype = np.float64)
        self._prior = np.zeros(len(self._classes), dtype = np.float64)
        
        for idx, class_label in enumerate(self._classes):
            
            X_c = X[class_label == Y]
            
            self._mean[idx, :] = np.mean(X_c, axis = 0)
            self._var[idx, :] = np.var(X_c, axis = 0)
            self._prior[idx] = X_c.shape[0]/float(samples)
            
    def predict(self, X):
        
        posterior = [ self._predict(x) for x in X]
        
        return np.array(posterior)
    
    def _predict(self, x):
        
        posterior = []
        
        # calculate posterior probability for each class
        for idx, class_label in enumerate(self._classes):
            
            prior = np.log(self._prior[idx])  
            likelihood = np.sum(np.log(self._find_likelihood(idx, x)))    
            poster = prior + likelihood       
            posterior.append(poster)
            
        # return class with highest posterior probability
        return self._classes[np.argmax(posterior)]
        
    def _find_likelihood(self, index, x):
        
        numerator = np.exp( - (x - self._mean[index])**2/(2*self._var[index]))
        denominator = np.sqrt(2*np.pi*self._var[index])
        
        likelihood = numerator/denominator
        
        return likelihood

#%%
if __name__=="__main__":
    
    X, Y = make_classification(n_samples = 1000, 
                               n_features = 10, 
                               n_classes = 2, 
                               random_state = (1234), 
                               shuffle = (True))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                        test_size = 0.2, 
                                                        random_state = 1234, 
                                                        shuffle = True)
    
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    
    #%%
    #From scratch
    
    obj = NaiveBayes()
    obj.fit(X_train, Y_train)    
    Y_hat = obj.predict(X_test)    
    print(np.mean((Y_hat == Y_test))*100)
    
    #%%
    #Using Inbuilt Function
    bayes = GaussianNB()
    bayes.fit(X_train, Y_train)
    Y_hat = bayes.predict(X_test)
    print(np.mean(Y_test == Y_hat)*100)
