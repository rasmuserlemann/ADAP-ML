## Script for performing PLS-DA and visualizing the results
# Data should be found in this repository under:
#              'data/SCLC_study_output_filtered_2.csv'
#
# PLS implementation form scikit-learn.cross_decomposition

# Created on January 17, 2020
# Author: Chris Avery

#import preprocess as pre
import numpy as np
import loadTestData as load_data
import sklearn.preprocessing as pre
from sklearn.cross_decomposition import PLSRegression as PLS
from matplotlib import pyplot as plt

##################### METHODS #####################
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
            plt.xlabel("PLS Component "+str(i+1))
            plt.ylabel("PLS Component "+str(j+1))
            
    plt.show()
    


##################### MAIN CODE #####################
#### Load data into numpy array'
# Keep pandas just for conveinience right now
data = load_data.loadDataPandas('../data/SCLC_study_output_filtered_2.csv')
d = data.to_numpy()
var_index = data.columns.values.tolist()

# vector of class responses associated with data
resp = load_data.getResponseMatrix2D()

#### Create object to normalize and un-normalize data
norm_trans = pre.StandardScaler().fit(d)
data_norm = norm_trans.transform(d)
#data_norm, norm_trans = pre.mean_center(d) 
#In-built preprocessing method - TBD

#### Fit a Partial Least Squaresn
pls = PLS().fit(data_norm, resp)
pls_trans = pls.transform(data_norm)

plotProjectionScatterMultiClass(pls_trans, resp, 2)