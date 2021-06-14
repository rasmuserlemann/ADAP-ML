# SOM attempt. No class.

import numpy as np
import loadTestData as ld
import sklearn.preprocessing as pre
from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

######### METHODS
def distance(x,y):
    delta = x - y
    d = np.sum(delta**2)
    return d

def get_dist(sample, weights):
    x, y, M = weights.shape
    dist_mat = np.ndarray(shape=[x, y])
    
    for i in range(x):
        for j in range(y):
            w = weights[i,j,:]
            dist_mat[i,j] = distance(sample, w)
    
    #print();print(dist_mat);print()
    return dist_mat

def get_closest_node(dist):
    inx_min = np.unravel_index(np.argmin(dist), shape=dist.shape)
    min_dist = dist[inx_min]
    return inx_min, min_dist

def get_sample_hits(data, weights):
    m, n = data.shape
    count_mat = np.zeros(shape=[num_node_x, num_node_y])
    for i in range(m):
        dist = get_dist(data[i,:], weights)
        x_min, min_dist = get_closest_node(dist)
        count_mat[x_min[0], x_min[1]] += 1
        
    print(count_mat)
    return count_mat


######### Main Code
# Load Data
do = "spectra"
if(do == "spectra"):
    data = ld.loadDataNumpy('../../data/SCLC_study_output_filtered_2.csv')
    classes = {"no-cancer":data[0:19,:], "cancer":data[20:39,:]}
elif(do == "fisher"):
    data = ld.loadDataNumpy('../../data/fisher_iris.csv')
    classes = {"setosa":data[0:49,:], "versicolor":data[50:99,:],
               "verginica":data[100:149,:]}

# Parameters
M, n = data.shape
num_node_x = 1
num_node_y = 2
lr = .01
band_width = 1
lr_soft = .5

norm = pre.StandardScaler().fit(data)
data_norm = norm.transform(data)

mumu = np.mean(np.abs(data_norm))
#data_norm = np.random.uniform(size=(M,n))

keys = list(classes.keys());
for key in keys:
    tmp = classes[key]
    tmp = norm.transform(tmp)
    #print(tmp)
    classes[key] = tmp

# Initialize weights (STEP 1)
W = mumu*np.random.uniform( size=(num_node_x,num_node_y,n) )

#W = mumu*W
print("Initial Clustering:")
initial_cluster = get_sample_hits(data_norm, W)

# Training
iter_num = 100000;
for k in range(iter_num):
    # STEP 2
    samp = data_norm[randint(0,M-1),:]
    
    # STEP 3
    dist = get_dist(samp, W)
    inx_min, min_dist = get_closest_node(dist)
    
    df = np.ndarray(shape=[num_node_x, num_node_y])
    for i in range(num_node_x):
        for j in range(num_node_y):
            d = np.abs(i-inx_min[0])**2 + np.abs(j-inx_min[1])**2
            df[i,j] = np.exp(-1*k/(2*band_width**2))
    #print();print(df);print()
    
    for i in range(num_node_x):
        for j in range(num_node_y):
            lr_tmp = lr*np.exp(-1*k/lr_soft)
            
            delt = df[i,j]*lr_tmp*(W[i,j,:]-samp)
            new_W = W[i,j,:] + delt
            W[i,j,:] = new_W
    #get_sample_hits(data_norm, W)
    
print();print("Final Clustering:")
final_cluster = get_sample_hits(data_norm, W)

# Get Projection of each Class
for key in keys:
    print();print("Clustering for "+key+":")
    cluster_cluster = get_sample_hits(classes[key], W)

# PLOTTING
X = np.arange(0, num_node_x, 1)
Y = np.arange(0, num_node_y, 1)
X, Y = np.meshgrid(X, Y)


#fig = plt.figure()
#ax = fig.gca(projection='3d')
#fig = plt.figure(); ax.plot_surface(X, Y, W[:,:,0],cmap='viridis', edgecolor='none')
#fig = plt.figure(); ax.plot_surface(X, Y, W[:,:,1],cmap='viridis', edgecolor='none')
#fig = plt.figure(); ax.plot_surface(X, Y, W[:,:,2],cmap='viridis', edgecolor='none')
#fig = plt.figure(); ax.plot_surface(X, Y, W[:,:,3],cmap='viridis', edgecolor='none')

#fig = plt.figure(); plt.imshow(W[:,:,0], cmap='hot', interpolation='nearest');
#fig = plt.figure(); plt.imshow(W[:,:,1], cmap='hot', interpolation='nearest');
#fig = plt.figure(); plt.imshow(W[:,:,2], cmap='hot', interpolation='nearest');
#fig = plt.figure(); plt.imshow(W[:,:,3], cmap='hot', interpolation='nearest');

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#fig = plt.figure(); ax.plot_surface(X, Y, W[:,:,0],cmap='viridis', edgecolor='none')