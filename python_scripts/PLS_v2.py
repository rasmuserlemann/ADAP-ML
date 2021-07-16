## Script for performing PLS-DA and visualizing the results
# Data should be found in this repository under:
#              'data/SCLC_study_output_filtered_2.csv'
#
# PLS implementation form scikit-learn.cross_decomposition

# Created on February 15, 2020
# Author: Chris Avery

#import preprocess as pre
import numpy as np
import loadTestData as load_data
import sklearn.preprocessing as pre
from sklearn.cross_decomposition import PLSRegression as PLS
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

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
    
def getVIP(pls_model):
    T = pls_model.x_scores_
    W = pls_model.x_weights_
    B = pls_model.x_rotations_
    
    p = pls_model.x_weights_.shape[0]
    A = pls_model.x_weights_.shape[1]
    n = pls_model.x_scores_.shape[0]
    
    vip = np.ndarray(p)
    for j in range(p):
        tmp1 = 0
        tmp2 = 0
        for a in range(A):
            tmp1 = tmp1 + W[j,a]**2 * np.linalg.norm(B[:,a],2) * np.linalg.norm(T[:,a],2)
            tmp2 = tmp2 + np.linalg.norm(B[:,a],2) * np.linalg.norm(T[:,a],2)
        
        vip[j] = tmp1/tmp2
    
    return vip

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
crossval = KFold(n_splits=5)
cv_2 = crossval.split(data_norm, resp)

cv_accuracy = np.ndarray(5)
models = list()
vip = np.ndarray((5, 19))
iter = 0
for (train, test) in cv_2:
    train = np.array(train)
    test = np.array(test)
    train_data = data_norm[train,:]
    train_resp = resp[train,:]
    test_data = data_norm[test,:]
    test_resp = resp[test,:]
    
    models.append(PLS().fit(train_data, train_resp))
    pls_comps = models.__getitem__(iter).transform(train_data)
    
    testing = models.__getitem__(iter).predict(test_data)
    
    tmp = np.sqrt(np.mean((testing - resp[test])**2))
    cv_accuracy[iter] = tmp
    #print(testing)
    #print(resp[test])
    #print(tmp)
    
    vip[iter, :] = getVIP(models.__getitem__(iter))
    #print(vip)
    
    iter = iter + 1
    
for i in range(5):
    plot = plt.figure
    plt.plot(vip[i, :])



#plotProjectionScatterMultiClass(pls_trans, resp, 2)
