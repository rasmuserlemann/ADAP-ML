## Script for performing Agglomerative Hierarcical clustering and visualizing the results
# Data should be found in this repository under:
#              'data/SCLC_study_output_filtered_2.csv'
#
# Agglomerative clustering implementation form scikit-learn.cluster

# Created on January 21, 2020
# Author: Chris Avery

import numpy as np
import loadTestData as load
from matplotlib import pyplot as plt
from sklearn import cluster as clst
from sklearn import preprocessing as pre
from scipy.cluster.hierarchy import dendrogram

##################### METHODS #####################
def plot_dendrogram(model, sample_labels):
    # Authors: Mathew Kallada
    # Link: https://github.com/scikit-learn/scikit-learn/blob/
    #               70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/
    #               plot_hierarchical_clustering_dendrogram.py
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(14, 9))
    dendrogram(linkage_matrix, labels=sample_labels)


##################### MAIN CODE #####################
## Load data
data = load.loadDataPandas('../data/SCLC_study_output_filtered_2.csv')
var_names = data.columns.values.tolist()
samp_names = data.index.values.tolist()
data = data.to_numpy()
resp = load.getResponseMatrix1D()

# Normalize data
norm = pre.StandardScaler().fit(data)
data_norm = norm.transform(data)

## Set up model for sample clustering
n_clusters_samp=2
aff_samp='cosine' #distance metric: {euclidean, manhattan, cosine, precomputted}
link_samp='average' #cluster linkage metric: {average, complete, single, ward}

hclust_samp = clst.AgglomerativeClustering(n_clusters=n_clusters_samp,
                                      affinity=aff_samp,
                                      linkage=link_samp)
hclust_samp.fit(data_norm)

plot_dendrogram(hclust_samp, samp_names)


## Set up model for variable clustering
n_clusters_var=2
aff_var='cosine' #distance metric: {euclidean, manhattan, cosine, precomputted}
link_var='average' #cluster linkage metric: {average, complete, single, ward}

hclust_var = clst.AgglomerativeClustering(n_clusters=n_clusters_var,
                                      affinity=aff_var,
                                      linkage=link_var)
hclust_var.fit(data_norm.transpose())

plot_dendrogram(hclust_var, var_names)