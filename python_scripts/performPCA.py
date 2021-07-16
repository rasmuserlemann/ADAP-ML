## Script for performing PCA and visualizing the results
# Data should be found in this repository under:
#              'data/SCLC_study_output_filtered_2.csv'
#
# PCA implementation form scikit-learn

# Created on January 16, 2020
# Author: Chris Avery

import numpy as np
import loadTestData as load_data
import sklearn.preprocessing as pre
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

##################### METHODS #####################
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

##################### MAIN CODE #####################
#### Load data into numpy array
data = load_data.loadDataPandas('../data/SCLC_study_output_filtered_2.csv')
d = data.to_numpy()
resp = load_data.getResponseMatrix2D()

#### Create object to normalize and un-normalize data
norm_trans = pre.StandardScaler().fit(d)
data_norm = norm_trans.transform(d)

#### Fit a Principal Component Analysis
pca = PCA().fit(data_norm)
pc = pca.transform(data_norm)

#### Get and plot scree and cumulative variance of components
plotScree(pca.explained_variance_, len(pca.explained_variance_))

#### Plot PC feature projections
plotProjectionScatterMultiClass(pc, resp, 5)

#### Plot Eigenvector loadings
varis = data.columns.values.tolist()
plotVectorLoadings(pca.components_, varis, 5)

#### Plot Eigenvector Loading Scatter Plots
plotLoadingsScatter(pca.components_)