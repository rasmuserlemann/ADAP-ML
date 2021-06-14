## Script for Generating a Self-Organizing Map for Clustering analysi
# Data should be found in this repository under:
#              'data/SCLC_study_output_filtered_2.csv'
#
# SOM implementation using tensorflow

# Created on January 21, 2020
# Author: Chris Avery

import tensorflow as tf
import loadTestData as ld
import numpy as np
import sklearn.preprocessing as pre

class SOM:
    # Constructor
    def __init__(self, x_grid, y_grid, data, band_width, lr=1):
        self.x_grid = x_grid # up/down
        self.y_grid = y_grid # left/right
        self.lr = lr
        self.band_width = band_width
        self.tr_data = data
        self.feat_dim = self.tr_data.shape[1]
        mu = np.mean(data)
        self.W = tf.random.uniform(shape=[x_grid, y_grid, self.feat_dim])
    
    def d_cost(self, x, w):
        dc = x - w
        return dc
    
    def cost(self, x, w):
        c = .5*(np.linalg.norm(self.d_cost(x, w))**2)
        return c
    
    def get_dist_mat(self, x):
        dist = tf.Variable( tf.zeros([self.x_grid, self.y_grid]) )
        for i in range(self.x_grid):
            for j in range(self.y_grid):
                w = self.W[i,j,:]
                dist[i,j].assign(self.cost(x, w))
        return dist
    
    def update_W(self, x_k, iter):
        dist = self.get_dist_mat(x_k)
        delt = tf.Variable( tf.zeros([self.x_grid, self.y_grid, self.feat_dim]) )
        for i in range(self.x_grid):
            for j in range(self.y_grid):
                lr = self.lr#*np.exp(-1*iter/.1) # Exponential Decay
                df = np.exp(-1*dist[i,j]/(2*self.band_width**2))
                w = self.W[i,j,:]
                
                tmp = tf.math.add(self.W[i,j,:], lr*df*self.d_cost(x_k, w))
                delt[i,j].assign(tmp)
        
        self.W = tf.add(self.W, delt)
        print(self.W)
    
    def fit(self, data, epochs):
        for epoch in range(epochs):
            for k in range(data.shape[0]):
                self.update_W(data[k,:], epoch)    
            print(self.map_data(data))
        return self.W
    
    def map_data(self, data):
        for k in range(data.shape[0]):
            d = self.get_dist_mat(data[k,:])

### Main Code
data = ld.loadDataPandas('../../data/SCLC_study_output_filtered_2.csv')
var_names = data.columns.values.tolist()
samp_names = data.index.values.tolist()
data = data.to_numpy()

norm_trans = pre.StandardScaler().fit(data)
data_norm = norm_trans.transform(data)

som_model = SOM(5, 5, data, band_width=np.mean(data_norm))
som_model.fit(data_norm, 1)