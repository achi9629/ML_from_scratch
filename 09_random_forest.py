# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:53:51 2021

@author: HP
"""
#%%
from decision_tree import DT
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def boot_strap_sample(X, Y):
    samples = X.shape[0]
    idxs = np.random.choice(samples, samples, replace = True)
    return X[idxs], Y[idxs]

def most_common_label(Y):
    counter = Counter(Y)
    most_common = counter.most_common(1)
    return most_common[0][0]


class RF:
    
    def __init__(self, n_trees = 5, min_samples_split = 2, max_depth = 100, n_feats = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
        
    def fit(self, X, Y):
        self.trees = []
        
        for tree in range(self.n_trees):
            
            tree = DT(min_samples_split = self.min_samples_split, 
                      max_depth = self.max_depth, 
                      n_feats = self.n_feats)
            
            X_sample, Y_sample = boot_strap_sample(X, Y)
            tree.fit(X_sample, Y_sample)
            self.trees.append(tree)
        

    def predict(self, X):
        
        tree_preds = [ tree.predict(X) for tree in self.trees]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        Y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return Y_pred

#%%    
if __name__ == "__main__":
    
    bc = datasets.load_breast_cancer()
    data, target = bc.data, bc.target
    
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, 
                                                        test_size = 0.2, 
                                                        random_state = 1234, 
                                                        shuffle = True)
    #%%
    #From Scratch
    obj = RF(n_trees = 3)
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_test)
    print('Random Forest Accuracy: ',np.mean(Y_hat == Y_test)*100)
    
    
    #%%
    #using Inbuilt Function
    forest = RandomForestClassifier(n_estimators = 3, criterion = 'entropy')
    forest.fit(X_train, Y_train)
    Y_hat = forest.predict(X_test)    
    print('Accuracy: ',np.mean(Y_hat == Y_test)*100)
