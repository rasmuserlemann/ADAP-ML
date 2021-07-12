import nbformat as nbf
import os
import time
#import config
import uuid
import sys
from nbconvert.preprocessors import ExecutePreprocessor

#Read in data location as an argument in terminal
datafile = str(sys.argv[1])

nb = nbf.v4.new_notebook()
title = """\
% ADAP-ML Report
"""

stattext = """\
# Statistics

T-test
"""


statcode = """\
import modules.adapml_data as adapml_data
import modules.adapml_classification as adapml_classification
import modules.adapml_clustering as adapml_clustering
import modules.adapml_chemometrics as adapml_chemometrics
import modules.adapml_statistics as adapml_statistics
import numpy as np
import modules.loadTestData as load_data
import sklearn.preprocessing as pre
from sklearn.cross_decomposition import PLSRegression as PLS
from matplotlib import pyplot as plt
from sklearn import cluster as clst
from scipy.cluster.hierarchy import dendrogram

import os

reldir = os.getcwd()
path_to_data = os.path.join(reldir, '..', 'data', '{}')

data = adapml_data.DataImport(path_to_data)

response1D = data.getResponseNew()
#response1D = adapml_data.DataImport.getResponse(path_to_data)
response2D = adapml_data.DataImport.getDummyResponse(response1D)

variables = data.getVariableNames()
samples = data.getSampleNames()

t_test = adapml_statistics.Statistics(data.data, 'anova', response1D)
t_test.plot_logp_values(variables)
t_test.plot_volcano_t(variables)


""".format(datafile)

dimtext = """\
# Dimension-Reduction

PCA, LDA
"""

dimcode = """\
data.normalizeData("autoscale")

pca = adapml_chemometrics.Chemometrics(data.data, "pca", response1D)
lda = adapml_chemometrics.Chemometrics(data.data, "lda", response1D) # Also Predicts

print("PCA Projections");pca.plotProjectionScatterMultiClass(2, labels=["Healthy", "Not Healthy"])
print("LDA Projections");lda.plotProjectionScatterMultiClass(1, labels=["Healthy", "Not Healthy"])

print("PCA Vectors"); pca.plotVectorLoadings(variables, 1)
print("LDA Vectors"); lda.plotVectorLoadings(variables, 1)
"""

clustertext = """\
# Clustering

K-means, hierarchical,

"""

clustercode = """\
kmeans_cluster = adapml_clustering.Clustering(data.data, 'kmeans', 3)
kmeans_cluster.getClusterResults(samples)

hierarchical_cluster = adapml_clustering.Clustering(data.data, 'hierarchical', 3)
hierarchical_cluster.getClusterResults(samples)
hierarchical_cluster.plot_dendrogram(samples)


"""

classiftext = """\
# Classification

PLS-DA

"""

classifcode = """\
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

data = load_data.loadDataPandas(path_to_data)
d = data.to_numpy()
var_index = data.columns.values.tolist()

resp = load_data.getResponseMatrix2D()

norm_trans = pre.StandardScaler().fit(d)
data_norm = norm_trans.transform(d)
#data_norm, norm_trans = pre.mean_center(d)
#In-built preprocessing method - TBD

pls = PLS().fit(data_norm, resp)
pls_trans = pls.transform(data_norm)

plotProjectionScatterMultiClass(pls_trans, resp, 2)


data = adapml_data.DataImport(path_to_data)
svm = adapml_classification.Classification(data.data, response1D, 'svm', .75, kfolds=3)
rnf = adapml_classification.Classification(data.data, response1D, 'randomforest', .75, kfolds=3)


adapml_classification.print_model_stats(svm, "SVM")
adapml_classification.print_model_stats(rnf, "RF")
"""

regressiontext = """\
# Regression

Linear regression

"""

regressioncode = """\

"""

nb['cells'] = [nbf.v4.new_markdown_cell(title),
               nbf.v4.new_markdown_cell(stattext), nbf.v4.new_code_cell(statcode),
               nbf.v4.new_markdown_cell(dimtext), nbf.v4.new_code_cell(dimcode),
               nbf.v4.new_markdown_cell(clustertext), nbf.v4.new_code_cell(clustercode),
               nbf.v4.new_markdown_cell(classiftext), nbf.v4.new_code_cell(classifcode),
               nbf.v4.new_markdown_cell(regressiontext), nbf.v4.new_code_cell(regressioncode)]

#New folder with a unique name
folder_name = 'Analysis_' + time.strftime("%Y_%m_%d_%H_%M_%S") + "_" + str(uuid.uuid4())
os.mkdir(folder_name)

#Make src folder and copy the python files
os.mkdir(folder_name + '/src')
os.mkdir(folder_name + '/src/modules')
os.system('cp modules/adapml_chemometrics.py ' + folder_name + '/src/modules')
os.system('cp modules/adapml_clustering.py ' + folder_name + '/src/modules')
os.system('cp modules/adapml_classification.py ' + folder_name + '/src/modules')
os.system('cp modules/adapml_statistics.py ' + folder_name + '/src/modules')
os.system('cp modules/loadTestData.py ' + folder_name + '/src/modules')
os.system('cp modules/OPLS.py ' + folder_name + '/src/modules')
os.system('cp modules/adapml_data.py ' + folder_name + '/src/modules')

os.mkdir(folder_name + '/results')

#Copy the data
os.system('cp -R data ' + folder_name )
with open(folder_name + '/src/report.ipynb', 'w') as f:
    nbf.write(nb, f)

with open(folder_name + '/src/report.ipynb') as f:
    nb = nbf.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {})
with open(folder_name + '/src/report_executed.ipynb', 'wt') as f:
    nbf.write(nb, f)

os.system('jupyter nbconvert --output-dir=' + folder_name + '/results --to PDF --output report.pdf --no-input ' + folder_name + '/src/report_executed.ipynb')
os.system('jupyter nbconvert --output-dir=' + folder_name + '/results --to PDF --output report_code.pdf ' + folder_name + '/src/report_executed.ipynb')

