import nbformat as nbf
import os
import time
#import config
import uuid
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbf.v4.new_notebook()
text1 = """\
# ADAP-ML Report

T-test and volcano plot
"""

code1 = """\
import adapml_data
import adapml_classification
import adapml_chemometrics
import adapml_statistics
import numpy as np
import loadTestData as load_data
import sklearn.preprocessing as pre
from sklearn.cross_decomposition import PLSRegression as PLS
from matplotlib import pyplot as plt
import os


reldir = os.getcwd()
path_to_data = os.path.join(reldir, '..', 'data', 'SCLC_study_output_filtered_2.csv')
path_to_resp = os.path.join(reldir, '..', 'data', 'SCLC_study_responses_2.csv')

data = adapml_data.DataImport(path_to_data)
response1D = adapml_data.DataImport.getResponse(path_to_resp)
response2D = adapml_data.DataImport.getDummyResponse(response1D)

variables = data.getVariableNames()
samples = data.getSampleNames()

t_test = adapml_statistics.Statistics(data.data, 'anova', response1D)
t_test.plot_logp_values(variables)
t_test.plot_volcano_t(variables)


"""
text2 = """\
# PCA
blabla"""

code2 = """\
data.normalizeData("autoscale")

pca = adapml_chemometrics.Chemometrics(data.data, "pca", response1D)

print("PCA Projections");pca.plotProjectionScatterMultiClass(2, labels=["Healthy", "Not Healthy"])

"""

text3 = """\
# PLS-DA
"""

code3 = """\
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
"""

nb['cells'] = [nbf.v4.new_markdown_cell(text1),
               nbf.v4.new_code_cell(code1),
               nbf.v4.new_markdown_cell(text2),
               nbf.v4.new_code_cell(code2),
               nbf.v4.new_markdown_cell(text3),
               nbf.v4.new_code_cell(code3)]

#New folder with a unique name
folder_name = 'Analysis_' + time.strftime("%Y_%m_%d_%H_%M_%S") + "_" + str(uuid.uuid4())
os.mkdir(folder_name)

#Make src folder and copy the python files
os.mkdir(folder_name + '/src')
os.system('cp adapml_chemometrics.py ' + folder_name + '/src')
os.system('cp adapml_clustering.py ' + folder_name + '/src')
os.system('cp adapml_classification.py ' + folder_name + '/src')
os.system('cp adapml_statistics.py ' + folder_name + '/src')
os.system('cp loadTestData.py ' + folder_name + '/src')
os.system('cp OPLS.py ' + folder_name + '/src')
os.system('cp adapml_data.py ' + folder_name + '/src')

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

