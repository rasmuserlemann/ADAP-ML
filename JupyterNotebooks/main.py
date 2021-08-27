import nbformat as nbf
import os
import time
import uuid
import sys
import modules.ConstructJupyterNotebook as Jup
import modules.config as config
from nbconvert.preprocessors import ExecutePreprocessor

#Read in data location as an argument in terminal
datafile = str(sys.argv[1])

nb = nbf.v4.new_notebook()

#Take True/False values from the configuration file
Jup.volcano_text = Jup.volcano_text * config.volcano
Jup.volcano_code = Jup.volcano_code * config.volcano

#Combine jupyter notebook text and code cells
nb['cells'] = [
               nbf.v4.new_markdown_cell(Jup.statistics_text),
               nbf.v4.new_markdown_cell(Jup.t_test_text), nbf.v4.new_code_cell(Jup.t_test_code(datafile)),
               nbf.v4.new_markdown_cell(Jup.volcano_text), nbf.v4.new_code_cell(Jup.volcano_code),
               nbf.v4.new_markdown_cell(Jup.bonferroni_text), nbf.v4.new_code_cell(Jup.bonferroni_code),
               nbf.v4.new_markdown_cell(Jup.dim_text),
               nbf.v4.new_markdown_cell(Jup.pca_text), nbf.v4.new_code_cell(Jup.pca_code),
               nbf.v4.new_markdown_cell(Jup.lda_text), nbf.v4.new_code_cell(Jup.lda_code),
               nbf.v4.new_markdown_cell(Jup.cluster_text), nbf.v4.new_code_cell(Jup.cluster_code),
               nbf.v4.new_markdown_cell(Jup.kmeans_text), nbf.v4.new_code_cell(Jup.kmeans_code),
               nbf.v4.new_markdown_cell(Jup.birch_text), nbf.v4.new_code_cell(Jup.birch_code),
               nbf.v4.new_markdown_cell(Jup.dbscan_text), nbf.v4.new_code_cell(Jup.dbscan_code),
               nbf.v4.new_markdown_cell(Jup.meanshift_text), nbf.v4.new_code_cell(Jup.meanshift_code),
               nbf.v4.new_markdown_cell(Jup.gaussian_text), nbf.v4.new_code_cell(Jup.gaussian_code),
               nbf.v4.new_markdown_cell(Jup.hier_text), nbf.v4.new_code_cell(Jup.hier_code),
               nbf.v4.new_markdown_cell(Jup.classif_text),
               nbf.v4.new_markdown_cell(Jup.plsda_text), nbf.v4.new_code_cell(Jup.plsda_code),
               nbf.v4.new_markdown_cell(Jup.neural_text), nbf.v4.new_code_cell(Jup.neural_code),
               nbf.v4.new_markdown_cell(Jup.svm_text), nbf.v4.new_code_cell(Jup.svm_code),
               nbf.v4.new_markdown_cell(Jup.rf_text), nbf.v4.new_code_cell(Jup.rf_code),
               nbf.v4.new_markdown_cell(Jup.logistic_text), nbf.v4.new_code_cell(Jup.logistic_code),
               nbf.v4.new_markdown_cell(Jup.regression_text),
               nbf.v4.new_markdown_cell(Jup.linreg_text), nbf.v4.new_code_cell(Jup.linreg_code)]

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

os.system('jupyter nbconvert --output-dir=' + folder_name + '/results --to PDF --template=./revtex.tplx --output report.pdf --no-input ' + folder_name + '/src/report_executed.ipynb' )
os.system('jupyter nbconvert --output-dir=' + folder_name + '/results --to PDF --template=./revtex.tplx --output report_code.pdf ' + folder_name + '/src/report_executed.ipynb' )

