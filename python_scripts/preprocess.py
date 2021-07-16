# Script for implementing several useful preprocessing methods

# Included Methods:
#       Sphering
#       Mean-Centering

# Created on January 16, 2020
# Author: Chris Avery

from sklearn import preprocessing as pre
import numpy as np

def sphere_data(data):
    transformer = pre.StandardScaler().fit(data) #Optional output
    data_trans = transformer.transform(data)
    
    return data_trans, data_trans

def mean_center(data):
    mu = np.mean(data, axis=0)
    data_trans = data - mu
    
    return data_trans, mu