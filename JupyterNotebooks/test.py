import os
import adapml_data
import adapml_classification
import adapml_chemometrics
import adapml_statistics
import loadTestData as load_data
import sklearn.preprocessing as pre
from sklearn.cross_decomposition import PLSRegression as PLS
from matplotlib import pyplot as plt

reldir = os.getcwd()
path_to_data = os.path.join(reldir, '..', 'data', 'SCLC_study_output_filtered_2.csv')
path_to_resp = os.path.join(reldir, '..', 'data', 'SCLC_study_responses_2.csv')

data = adapml_data.DataImport(path_to_data)
response1D = adapml_data.DataImport.getResponse(path_to_resp)
response2D = adapml_data.DataImport.getDummyResponse(response1D)

variables = data.getVariableNames()
samples = data.getSampleNames()

data.normalizeData("autoscale")

pca = adapml_chemometrics.Chemometrics(data.data, "pca", response1D)
pls = adapml_chemometrics.Chemometrics(data.data, "pls-da", response1D, kfolds=10, num_comp=2) # Also Predicts
opls = adapml_chemometrics.Chemometrics(data.data, "opls", response1D, kfolds=10, num_comp=2, opls_comp=16) # Also Predicts
lda = adapml_chemometrics.Chemometrics(data.data, "lda", response1D) # Also Predicts