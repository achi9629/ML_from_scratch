# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:50:00 2021

@author: HP
"""
#%%
from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def entropy(Y):
    
    hist = np.bincount(Y)
    prob = hist/len(Y)
    
    return -np.sum([p*np.log2(p) for p in prob if p > 0])

class Node:
    
    def __init__(self, features = None, threshold = None, left = None, right = None, *, value = None):
        self.features = features
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None
    
class DT:
    
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        
    def fit(self, X, Y):
        self.n_feats = X.shape[1] if not self.n_feats else min(X.shape[1], self.n_feats)
        
        self.root = self._grow_tree(X, Y)
        
    def _grow_tree(self, X, Y, depth = 0):
        
        samples, features = X.shape
        labels = len(np.unique(Y))
        
        if depth >= self.max_depth or labels < self.min_samples_split or labels == 1:
             leaf_value = self._most_common_label(Y)
             return Node(value = leaf_value)
        
        feat_idxs = np.random.choice(features, self.n_feats, replace = False)
        
        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, Y, feat_idxs)
        
        left_idxs, right_idxs = self._split(X[:,best_feat], best_thresh)
        
        left = self._grow_tree(X[left_idxs,:], Y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs,:], Y[right_idxs], depth + 1)
        
        return Node(best_feat, best_thresh, left, right)
        
    def _best_criteria(self, X, Y, feat_idxs):
        
        best_gain = -1
        split_idx, split_thresh = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                
                gain = self._info_gain(X_column, Y, threshold)
                
                if gain > best_gain :
                    best_gain = gain
                    split_thresh = threshold
                    split_idx = feat_idx
                    
        return split_idx, split_thresh
        
    def _info_gain(self, X_column, Y, split_thresh):
        
        # parent loss
        parent_entropy = entropy(Y)
        
        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # compute the weighted avg. of the loss for the children
        n = len(Y)
        
        n_l, n_r = len(left_idxs), len(right_idxs)
        
        e_l, e_r = entropy(Y[left_idxs]), entropy(Y[right_idxs])
        
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        
        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        
        return ig
        
        
    def _split(self, X_column, split_thresh):
        
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        
        return left_idxs, right_idxs
        
        
    def _most_common_label(self, Y):
        counter = Counter(Y)
        most_common = counter.most_common(1)
        return most_common[0][0]
        
    def predict(self, X):
        return np.array([ self.traverse(X_column,self.root) for X_column in X])
        
    def traverse(self, X_column, node):
        
        if node.is_leaf_node():
            return node.value
        
        if X_column[node.features] <= node.threshold:
            return self.traverse(X_column, node.left)
        return self.traverse(X_column, node.right)

#%%
if __name__=="__main__":
    
    bc = datasets.load_breast_cancer()
    
    data, target = bc.data, bc.target
    
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, 
                                                        test_size = 0.2, 
                                                        random_state = 1234, 
                                                        shuffle = True)
    
    #%%
    #From Scratch
    obj = DT(max_depth = 5)
    obj.fit(X_train, Y_train)
    Y_hat = obj.predict(X_test)
    print(np.mean(Y_hat == Y_test)*100)
    
    #%%
    #Using Inbuilt Function
    tree = DecisionTreeClassifier(max_depth = 5)
    tree.fit(X_train, Y_train)
    Y_hat = tree.predict(X_test)
    print(np.mean(Y_test == Y_hat)*100)