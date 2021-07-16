## Testing OPLS

import numpy as np
import adapml_data
import adapml_chemometrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

path_to_data = 'C:\\Users\\csa97\\Research\\Projects\\DuLab\\ADAP-ML\\adap-ml\\data\\SCLC_study_output_filtered_2.csv'
path_to_resp = 'C:\\Users\\csa97\\Research\\Projects\\DuLab\\ADAP-ML\\adap-ml\\data\\SCLC_study_responses_2.csv'
#path_to_data = 'C:\\Users\\csa97\\Research\\Projects\\DuLab\\ADAP-ML\\adap-ml\\data\\fisher_iris.csv'
#path_to_resp = 'C:\\Users\\csa97\\Research\\Projects\\DuLab\\ADAP-ML\\adap-ml\\data\\fisher_iris_resp.csv'

data = adapml_data.DataImport(path_to_data)
response1D = adapml_data.DataImport.getResponse(path_to_resp);
response2D = adapml_data.DataImport.getDummyResponse(response1D);

variables = data.getVariableNames()
samples = data.getSampleNames()

data.normalizeData('autoscale')

X_train, X_test, y_train, y_test = train_test_split(
        data.data, response1D, test_size=0.25, shuffle=True)

###PLS

pls = adapml_chemometrics.Chemometrics(data.data, "pls-da", response1D, kfolds=10,
                                        num_comp=3)

pls.plotProjectionScatterMultiClass(3, labels=["Healthy", "Not Healthy"])
print("PLS Vectors"); pls.plotVectorLoadings(variables, 1)
print("PLS"); pls.plotVIP(variables, 1)
print("PLS: R^2="+str(pls.R2)+" Q^2="+str(pls.Q2))


##OPLS

# opls = adapml_chemometrics.Chemometrics(data.data, "opls",
#                                         response1D, kfolds=10,
#                                         num_comp=3, opls_comp=16)
# opls.plotProjectionScatterMultiClass(2, labels=["Healthy", "Not Healthy"])
# print("OPLS Vectors"); opls.plotVectorLoadings(variables, 1)
# print("OPLS"); opls.plotVIP(variables, 1)

# print("OPLS: R^2="+str(opls.R2)+" Q^2="+str(opls.Q2))



## SVM
#parameters = [{'kernel':['linear'], 
#               'shrinking':[True, False]},
#               {'kernel':['rbf'], 
#                'gamma': ['scale', 'auto'], 
#                'shrinking':[True, False]}, 
#               {'kernel':['poly'], 
#                'degree':[2,3,4], 
#                'gamma': ['scale', 'auto'], 
#                'shrinking':[True, False]}]
#
#svm = GridSearchCV(SVC(), param_grid=parameters, cv=5)
#svm.fit(X_train, np.ravel(y_train))
#print(svm.best_params_)
#
#y_pred = svm.predict(X_test)
#
#acc = accuracy_score(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)



## MLP
#
#parameters_mlp = [{'solver':['adam'],
#                  'activation':['identity', 'logistic', 'tanh'],
#                  'learning_rate':['constant', 'invscaling', 'adaptive']},
#                 {'solver':['sgd'],
#                  'activation':['identity', 'logistic', 'tanh'],
#                 'learning_rate':['constant', 'invscaling', 'adaptive'],
#                 'momentum':[0.5, 0.7, 0.9, 0.99]}]
#
#mlp = GridSearchCV(MLPClassifier(max_iter=5000), parameters_mlp, 
#                  cv=2, error_score=np.nan, iid=False)
#mlp.fit(X_train, np.ravel(y_train))
#
#print(mlp.best_params_)



## Random Forest
#
#parameters_rf = [{'n_estimators':[10, 50, 100, 500, 1000],
#               'criterion':['gini','entropy']}]
#
#rf = GridSearchCV(RandomForestClassifier(), parameters_rf, cv=5, iid = False)
#rf.fit(X_train, np.ravel(y_train))