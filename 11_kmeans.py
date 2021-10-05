# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:10:42 2021

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KM:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [ [] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []


    def predict(self, X):
        
        self.X = X
        self.samples, self.features = X.shape

        # initialize centroids
        indices = np.random.choice(self.samples, self.K, replace = False)
        self.centroids = [self.X[index] for index in indices ]

        # Optimize clusters
        for _ in range(self.max_iters):
            
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            
            if self.plot_steps:
                self.plot()
                
            # Calculate new centroids from the clusters
            old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            if self._is_converged(old_centroids, self.centroids):
                break
            
            if self.plot_steps:
                self.plot()


        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        
        labels = np.empty(self.samples)
        # each sample will get the label of the cluster it was assigned to
        for cluster_idx, cluster in enumerate(clusters):           
            for idx in cluster:
                labels[idx] = cluster_idx
                
        return labels

    def _create_clusters(self, centroids):
        
        # Assign the samples to the closest centroids to create clusters
        clusters = [ [] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        
        # distance of the current sample to each centroid
        distances = [ euclidean_distance(sample, x) for x in centroids]
        closest_distance =  np.argmin(distances)
        return closest_distance

    def _get_centroids(self, clusters):
        
        centroids = np.zeros((self.K, self.features))
        
        # assign mean value of clusters to centroids
        for cluster_idx, cluster in enumerate(clusters):
            mean_centroids = np.mean(self.X[cluster], axis = 0)
            centroids[cluster_idx] = mean_centroids
        
        return centroids

    def _is_converged(self, centroids_old, centroids):
        
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[idx], centroids[idx]) for idx in range(self.K)]
        
        return sum(distances) == 0
    
    def plot(self):
        
        plt.subplots(figsize=(12, 8))
        for cluster_idx, cluster in enumerate(self.clusters):
            
            x, y = self.X[cluster].T
            
            plt.scatter(x, y)
            
        for x, y in self.centroids:
            
            plt.scatter(x, y, marker = '*', c = 'k')


#%%
if __name__=="__main__":
    
    X, Y = make_blobs(n_samples = 10000, n_features = 2, shuffle = True, random_state= 1234)
    
    #From Scratch
    obj = KM(K = 3, plot_steps= False)
    Y_pred = obj.predict(X)    
    obj.plot()
    
    
    #%%
    #Using Inbuilt Function
    means  = KMeans(n_clusters = 3)
    means.fit(X)
    Y_hat = means.labels_
    centroids = means.cluster_centers_
    
    plt.subplots(figsize = (12,8))
    plt.scatter(X[:,0], X[:,1], c = Y_hat)
    plt.scatter(centroids[:,0], centroids[:,1], marker = '*', c = 'w')
    plt.show()
    
    
