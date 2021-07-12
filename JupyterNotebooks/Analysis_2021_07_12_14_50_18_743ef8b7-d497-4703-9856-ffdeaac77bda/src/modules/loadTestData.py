## Script for loading in test data
# Data should be found in this repository under:
#              'data/SCLC_study_output_filtered_2.csv'

# Created on January 16, 2020
# Author: Chris Avery

import pandas as pd
import numpy as np

# Writen on Windows 10 OS, change \ to / for linux or mac
def loadDataPandas(path, cols2obs=False):
    df = pd.read_csv(path, index_col=0)
    
    #handle case where columns are observations
    if cols2obs:
        df = df.transpose()
        
    return df

def loadDataNumpy(path, cols2obs=False):
    df = loadDataPandas(path, cols2obs) #Use prior pandas function for ease
    numpy_array = df.to_numpy() #Exchange to numpy
    return numpy_array

def getResponseMatrix2D():
    resp1 = np.concatenate( (np.zeros([20, 1]), np.ones([20,1])), axis=0)
    resp2 = np.concatenate( (np.ones([20, 1]), np.zeros([20,1])), axis=0)
    resp = np.concatenate( (resp1, resp2), axis=1)
    
    return resp

def getResponseMatrix1D():
    resp1 = np.concatenate( (np.zeros([20, 1]), np.ones([20,1])), axis=0)
    return resp1

## Testing area
data = loadDataNumpy('../data/SCLC_study_output_filtered_2.csv')