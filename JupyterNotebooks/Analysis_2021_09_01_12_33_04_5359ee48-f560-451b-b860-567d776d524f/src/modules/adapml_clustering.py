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
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.spatial import distance

from kneed import KneeLocator


class Clustering:
    def __init__(self, data0, method0, num_clusters0):
        self.data = data0
        self.method = method0
        self.num_clusters = num_clusters0
        
        if (self.method == "kmeans"):
            self.cluster = self.kmeans_()
            self.sse = self.SSE(clst.KMeans, self.num_clusters)
            self.ssb = self.SSB(clst.KMeans, self.num_clusters)
        elif (self.method == "hierarchical"):
            self.cluster = self.hierarchical_()
        elif (self.method == "elbow"):
            self.clustnr = self.elbow_()
        elif (self.method == "silhouette"):
            self.clustnr = self.silhouette()
        elif (self.method == "birch"):
            self.cluster = self.birch_()
        elif (self.method == "gaussian"):
            self.cluster = self.gaussian_()
        elif (self.method == "meanshift"):
            self.cluster = self.meanshift_()
        elif (self.method == "dbscan"):
            self.cluster = self.dbscan_()
        else:
            print("Clustering Method "+self.method+" Not Available")
    def elbow_(self):
        distortions = []
        K = range(1,int(len(self.data)/2))
        for k in K:
            kmeanModel = clst.KMeans(n_clusters=k).fit(self.data)
            distortions.append(sum(np.min(cdist(self.data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / self.data.shape[0])
        kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')
        return(kn.knee)
            
    def kmeans_(self):
        clust = clst.KMeans(self.num_clusters).fit(self.data)   
        return clust
    
    def gaussian_(self):
        clust = mixture.GaussianMixture(n_components=self.num_clusters, covariance_type='full').fit(self.data)   
        return clust
    
    def meanshift_(self):
        clust = clst.MeanShift().fit(self.data)   
        return clust
    
    def dbscan_(self):
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        #find the max curvature
        #plt.plot(distances)
        #plt.show()
        #epshat = np.max(distances)
        clust = clst.DBSCAN(eps = 1000000, min_samples = 2).fit(self.data)
        return clust 
        
    def birch_(self):
        clust = clst.Birch(n_clusters=self.num_clusters).fit(self.data)
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
        if hasattr(self.cluster, 'labels_'):
            y_pred = self.cluster.labels_
        else:
            y_pred = self.cluster.predict(self.data)
        clusters = {}
        for i in range(self.num_clusters):
                where = [k for k in range(M) if i==y_pred[k]]
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
    
    def silhouette(self):
        cutoff = int(len(self.data)/4)
        Svec = []
        for clustnr in range(2, cutoff):
            clust = clst.KMeans(clustnr).fit(self.data)  
            clustlabels = np.array(clust.labels_)
            S = []
            for ind in range(len(self.data)):
                #calculate a(i)
                a_all = []
                inside = self.data[np.where(clustlabels==clustlabels[ind])]
                if (len(inside)==1):
                    S.append(0)
                    continue
                for datain in inside:
                    a_all.append(distance.euclidean(self.data[ind], datain))
                a = (1/(len(inside)-1))*np.mean(a_all)
                #calculate b(i)
                outside_b = []
                for outclust in range(clustnr):
                    if outclust == clustlabels[ind]:
                        continue
                    b_clust = []
                    outside_temp = self.data[np.where(clustlabels==outclust)]
                    for dataout in outside_temp:
                        b_clust =+ (1/len(outside_temp))*distance.euclidean(self.data[ind], dataout)
                    outside_b.append(b_clust)
                b = np.min(outside_b)
                #calculate s(i)
                S.append((b-a)/(max(a,b)))
            Svec.append(np.mean(S))
        return(Svec.index(max(Svec))+2)
    
    def silhouette2(self):
        cutoff = int(len(self.data)/4)
        Svec = []
        for clustnr in range(2, cutoff):
            clusterer = clst.KMeans(clustnr)
            preds = clusterer.fit_predict(self.data)
            score = silhouette_score(self.data, preds)
            Svec.append(score)
        return(Svec.index(max(Svec))+2)
    
    def SSE(self, alg, clustnr):
        clust = alg(clustnr).fit(self.data)  
        clustlabels = np.array(clust.labels_)
        S = 0
        for cl in range(np.max(clustlabels)):
            cent = clust.cluster_centers_[cl]
            inside = self.data[np.where(clustlabels==cl)]
            for el in inside:
                S =+ (distance.euclidean(cent, el))**2
        return S
        
    def SSB(self, alg, clustnr):
        clust = alg(clustnr).fit(self.data)  
        clustlabels = np.array(clust.labels_)
        m = np.mean(self.data)
        S = 0
        for cl in range(np.max(clustlabels)):
            cent = clust.cluster_centers_[cl]
            S =+ (distance.euclidean(cent, m))**2
            size = len(self.data[np.where(clustlabels==cl)])
            S =+ size*(distance.euclidean(cent, m))**2
        return S
    
    
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

'''
### Testing
import os      
os.chdir('/Users/rerleman/Documents/Git/adap-ml/JupyterNotebooks/modules')
import adapml_data
from sklearn.metrics import silhouette_samples, silhouette_score
##### TESTING CODE 1
reldir = os.getcwd()
path_to_data = os.path.join(reldir, '..', 'data', 'SCLC_study_output_filtered_2.csv')
from sklearn import datasets

data = adapml_data.DataImport(path_to_data)

samples = data.getSampleNames()
ward_cluster = Clustering(data.data, 'kmeans', 2)
print(ward_cluster.ssb)
'''
