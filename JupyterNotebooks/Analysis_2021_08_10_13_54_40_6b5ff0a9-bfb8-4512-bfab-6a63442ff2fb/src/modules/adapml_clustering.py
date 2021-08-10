# Machine Learning and Statistics Library for ADAP
# Focus: Clustering
# Author: Chris Avery
# Last Update: February 24 2020

#Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Scipy Libraries
import scipy.cluster.hierarchy as hier
#Scikit-Learn Libraries
import sklearn.cluster as clst

class Clustering:
    def __init__(self, data0, method0, num_clusters0):
        self.data = data0
        self.method = method0
        self.num_clusters = num_clusters0
        
        if (self.method == "kmeans"):
            self.cluster = self.kmeans_()
        elif (self.method == "hierarchical"):
            self.cluster = self.hierarchical_()
        elif (self.method == "dbscan"):
            self.cluster = self.dbscan_()
        else:
            print("Clustering Method "+self.method+" Not Available")
            
    def kmeans_(self):
        clust = clst.KMeans(self.num_clusters).fit(self.data)
        return clust
    
    def dbscan_(self):
        clust = clst.DBSCAN.fit(self.data)
        return clust 

    def hierarchical_(self):
        aff='cosine' #distance metric: {euclidean, manhattan, cosine, precomputted}
        link='average' #cluster linkage metric: {average, complete, single, ward}
        clust = clst.AgglomerativeClustering(n_clusters=self.num_clusters,
                                      affinity=aff,
                                      linkage=link)
        clust.fit(self.data)
        
        return clust
    
    ## Support Methods
    def getClusterResults(self, names):
        M = len(names)
        clusters = {}
        for i in range(self.num_clusters):
                where = [k for k in range(M) if i==self.cluster.labels_[k]]
                tmp = list()
                for k in where:
                    tmp.append(names[k])
                clusters[i] = tmp
    
        clusters_results = pd.DataFrame()
        cluster_names = {}
        for cluster in range(self.num_clusters):
            tmp = pd.DataFrame()
            tmp.insert(column=cluster, value=clusters[cluster], loc=0)
            clusters_results = pd.concat([clusters_results, tmp], ignore_index=True, axis=1)
            cluster_names[cluster] = "Cluster "+str(cluster+1)
    
        clusters_results.rename(columns=cluster_names, inplace=True)
    
        print(clusters_results)
    
    def plot_dendrogram(self, sample_labels):
        # Authors: Mathew Kallada (found online)
        # Link: https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/
        #               plot_hierarchical_clustering_dendrogram.py
        #
        # Children of hierarchical clustering
        children = self.cluster.children_
    
        # Distances between each pair of children
        # Since we don't have this information, we can use a uniform one for plotting
        distance = np.arange(children.shape[0])
    
        # The number of observations contained in each cluster level
        no_of_observations = np.arange(2, children.shape[0]+2)
    
        # Create linkage matrix and then plot the dendrogram
        linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    
        # Plot the corresponding dendrogram
        plt.figure(figsize=(14, 9))
        hier.dendrogram(linkage_matrix, labels=sample_labels)
 
