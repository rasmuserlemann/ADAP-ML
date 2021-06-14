## Script for performing k-means clustering and visualizing the results
# Data should be found in this repository under:
#              'data/SCLC_study_output_filtered_2.csv'
#
# k-means implementation form scikit-learn.cluster

# Created on January 20, 2020
# Author: Chris Avery

import numpy as np
import loadTestData as load
from matplotlib import pyplot as plt
from sklearn import cluster as clst
from sklearn import preprocessing as pre

##################### METHODS #####################
def find_index(array, target):
    inx = list()
    size = len(array)
    
    for i in range(size):
        if array[i] == target:
            inx.append(i)
            
    return inx
    

def getClusterAccuracy(act_resp, pred_resp, nCluster):
    len_act = len(act_resp)
    len_pred = len(pred_resp)
    
    if len_act != len_pred:
        print("Number of samples must be the same between actual and predicted responses!")
        return
    
    #pred_clusters = np.ndarray([nCluster, 1])
    #act_clusters = np.ndarray([nCluster, 1])
    pred_clusters = {}
    act_clusters = {}
    for i in range(nCluster):
        pred_clusters[i] = find_index(pred_resp, i)
        act_clusters[i] = find_index(act_resp, i)
        
    # Get union of clusters
    purity_matrix = np.ndarray([nCluster, nCluster])
    for i in range(nCluster):
        for j in range(nCluster):
            tmp = set(pred_clusters[i]) & set(act_clusters[j])
            purity_matrix[i,j] = len(tmp)
        
    return purity_matrix


##################### MAIN CODE #####################
#### Load data
data = load.loadDataPandas('../data/SCLC_study_output_filtered_2.csv')
varnames = data.columns.values.tolist()
data = data.to_numpy()
resp = load.getResponseMatrix1D()

#### Perform k-means
numClust = 2
k_means = clst.KMeans(numClust).fit(data)

#### Test Accuracy using "Cluster Purity?"
purity = getClusterAccuracy(resp, k_means.labels_, numClust)
print(purity)
