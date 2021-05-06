import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hier

# Clustering Methods
def getClusterResults(cluster_struct, names, n_clust):
    M = len(names)
    clusters = {}
    for i in range(n_clust):
            where = [k for k in range(M) if i==cluster_struct.labels_[k]]
            tmp = list()
            for k in where:
                tmp.append(names[k])
            clusters[i] = tmp

    clusters_results = pd.DataFrame()
    cluster_names = {}
    for cluster in range(n_clust):
        tmp = pd.DataFrame()
        tmp.insert(column=cluster, value=clusters[cluster], loc=0)
        clusters_results = pd.concat([clusters_results, tmp], ignore_index=True, axis=1)
        cluster_names[cluster] = "Cluster "+str(cluster+1)

    clusters_results.rename(columns=cluster_names, inplace=True)

    return clusters_results

def plot_dendrogram(model, sample_labels):
    # Authors: Mathew Kallada (found online)
    # Link: https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/
    #               plot_hierarchical_clustering_dendrogram.py
    #
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
    hier.dendrogram(linkage_matrix, labels=sample_labels)

# Chemometrics Methods
# Methods: PCA Visualization
def getCumVar(scree):
    N = len(scree)
    cumvar = np.empty([N]);
    for i in range(N):
        sum = 0
        for j in range(i):
            sum = sum + scree[j]
        cumvar[i] = sum

    return cumvar

def plotScree(scree, num_modes):
    sumvar = np.sum(scree)
    scree = scree[:num_modes]
    cumvar = getCumVar(scree)/sumvar
    dim = np.arange(1,len(scree)+1)

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_ylabel("Scree")
    ax1.plot(dim, scree, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("% Cumulative Variance")
    ax2.plot(dim, cumvar, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Scree Plot")
    plt.xlabel("Dimension Index")
    plt.xticks(dim)
    plt.show()

def plotProjectionScatter(pc):
    plt.figure(figsize=(24, 18))
    for i in range(5):
        mode_1 = i+1
        for j in range(5):
            mode_2 = j+1
            plt.subplot(5,5,5*(i) + j + 1)
            pc1 = pc[:,mode_1-1]
            pc2 = pc[:,mode_2-1]
            plt.scatter(pc1, pc2)
            plt.xlabel("PC "+str(mode_1))
            plt.ylabel("PC "+str(mode_2))
    plt.show()

def plotProjectionScatterMultiClass(pc, resp, num_var):
    plt.figure(figsize=(24, 18))

    for i in range(num_var):
        for j in range(num_var):
            plt.subplot(5,5,5*(i) + j + 1)
            for c in range(resp.shape[1]):
                inx = np.where(resp[:,c] == 1)[0]
                tmp = pc[inx,:]
                pc1 = tmp[:,i]
                pc2 = tmp[:,j]
                plt.scatter(pc1, pc2)
            plt.xlabel("PC "+str(i+1))
            plt.ylabel("PC "+str(j+1))

    plt.show()

def plotVectorLoadings(vectors, var_names, num_vectors):
    plt.figure(figsize=(12, 3*num_vectors)) #Adaptable size to number of vecs
    for i in range(num_vectors):
        plt.subplot(num_vectors,1,i+1)
        vec = np.power(vectors[i,:], 2) # Note the eigenvectors are the row space
        plt.plot(vec)
        plt.ylabel("Squared PC Mode "+str(i+1))
        plt.xticks(np.arange(len(var_names)), var_names)
    plt.xlabel("Variables")
    plt.show()

def plotLoadingsScatter(vectors):
    plt.figure(figsize=(24, 18))
    for i in range(5):
        mode_1 = i+1
        for j in range(5):
            mode_2 = j+1
            plt.subplot(5,5,5*(i) + j + 1)
            pc1 = vectors[mode_1-1,:]
            pc2 = vectors[mode_2-1,:]
            plt.scatter(pc1, pc2)
            plt.xlabel("PC "+str(mode_1))
            plt.ylabel("PC "+str(mode_2))
    plt.show()
