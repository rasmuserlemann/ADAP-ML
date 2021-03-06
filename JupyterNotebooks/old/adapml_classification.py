# Machine Learning and Statistics Library for ADAP
# Focus: Classification
# Author: Chris Avery
# Last Update: February 24 2020

#Basic Libraries
import numpy as np
import matplotlib.pyplot as plt
#Scikit-learn Libraries
from sklearn.metrics import r2_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#adapml Libraries
import adapml_chemometrics as chemo
import adapml_data
#other
import warnings
warnings.filterwarnings("error")

class Classification:
    def __init__(self, data0, resp0, method0, train_percent, **kwargs):
        self.data = data0
        self.method = method0
        self.resp = resp0
        self.train_inx, self.test_inx = self.getTrainTestData(train_percent)
        
        self.train_data = self.data[self.train_inx,:]
        self.train_resp = self.resp[self.train_inx,:]
        self.test_data = self.data[self.test_inx,:]
        self.test_resp = self.resp[self.test_inx,:]
        
        if ('kfolds' in kwargs):
            self.kfolds = kwargs.get('kfolds')
        else:
            self.kfolds = 3

        if(self.method == "qda"):                
            self.classifier = self.quadratic_discriminant()
            
        elif(self.method == "svm"):
            self.train_resp = np.ravel(self.train_resp)
            self.test_resp = np.ravel(self.test_resp)
            
            self.classifier = self.svm()
            
        elif(self.method == "neuralnet"):
            self.classifier = self.neuralnet()
        
        elif(self.method == "randomforest"):
            self.classifier = self.randomforest()
        
        else:
            print("Classification Method '"+self.method+"' Not Found!")
        
    def quadratic_discriminant(self):
        parameters = {'reg_param': [0, .1, .25, .4, .5, .6, .75, .9, 1]}
        qda = GridSearchCV( QDA(),
                           parameters,
                           cv=self.kfolds,
                           error_score=np.nan)
        qda.fit(self.train_data, np.ravel(self.train_resp))
        resp_pred = np.reshape(qda.predict(self.test_data), self.test_resp.shape)
        val_acc = np.sqrt(np.mean((resp_pred - self.test_resp)**2))
        best = qda
        best.validation_acc = val_acc
        return best
    
    def svm(self):
        parameters = [{'kernel':['linear'], 
                       'shrinking':[True, False]},
                       {'kernel':['rbf'], 
                        'gamma': ['scale', 'auto'], 
                        'shrinking':[True, False]}, 
                       {'kernel':['poly'], 
                        'degree':[2,3,4], 
                        'gamma': ['scale', 'auto'], 
                        'shrinking':[True, False]}]
        svm = GridSearchCV(SVC(),
                           parameters,
                           cv=self.kfolds)
        svm = GridSearchCV(SVC(), param_grid=parameters, cv=5)#, iid=False)
        svm.fit(self.train_data, self.train_resp)
        print("SVM Validated Parameters: ", svm.best_params_)
        
        y_pred_tr = svm.predict(self.train_data)
        y_pred = svm.predict(self.test_data)
        self.R2 = r2_score(self.train_resp, y_pred_tr)
        self.Q2 = r2_score(self.test_resp, y_pred)
        
        return svm
    
    def randomforest(self):
        parameters = [{'n_estimators':[10, 50, 100, 500, 1000],
               'criterion':['gini','entropy']}]
            
        rand_forest = GridSearchCV(RandomForestClassifier(),
                           parameters,
                           cv = self.kfolds,
                           error_score=np.nan)#, iid=False)
        rand_forest.fit(self.train_data, np.ravel(self.train_resp))
        print("Random Forest Validated Parameters: ", rand_forest.best_params_)

        y_pred_tr = rand_forest.predict(self.train_data)
        y_pred = rand_forest.predict(self.test_data)
        self.R2 = r2_score(self.train_resp, y_pred_tr)
        self.Q2 = r2_score(self.test_resp, y_pred)
       
        return rand_forest
    
    def neuralnet(self):
        parameters_mlp = [{'solver':['adam'],
                  'activation':['identity', 'logistic', 'tanh'],
                  'learning_rate':['constant', 'invscaling', 'adaptive']},
                  {'solver':['sgd'],
                  'activation':['identity', 'logistic', 'tanh'],
                  'learning_rate':['constant', 'invscaling', 'adaptive'],
                  'momentum':[0.5, 0.7, 0.9, 0.99]}]

        mlp = GridSearchCV(MLPClassifier(max_iter=5000), parameters_mlp, 
                   cv=self.kfolds, error_score=np.nan)#, iid=False)
        mlp.fit(self.train_data, np.ravel(self.train_resp))
        print("MLP Validated Parameters: ", mlp.best_params_)
        
        y_pred_tr = mlp.predict(self.train_data)
        y_pred = mlp.predict(self.test_data)
        self.R2 = r2_score(self.train_resp, y_pred_tr)
        self.Q2 = r2_score(self.test_resp, y_pred)
        
        return mlp
    
    ##### Support Methods #####
    def getTrainTestData(self, train_percent):
        n_samp, n_var = self.data.shape
        
        rand_inx = np.random.permutation(n_samp);
        tr_num = int(np.round(train_percent*n_samp));
        
        train_index = rand_inx[0:tr_num]
        test_index = rand_inx[(tr_num+1):(n_samp)]
        
        return train_index, test_index
    
    def softmaxClassify(model, data, num_class):
        print("Under Construction")
        
    def confusionMatrix(prediction, response):
        print("Under Construction")
        


def getQ2(y_test, y_pred):
    resp = y_test
    pred = y_pred
    
    PRESS = np.sum((resp - pred)**2)
        
    mu = np.mean(resp)
    TSS = np.sum((resp-mu)**2)
    
    #print("PRESS="+str(PRESS)); print("TSS="+str(TSS))

    Q2 = 1-(PRESS/TSS)
    
    return Q2

def print_model_stats(model, name):
    print(name+": R^2="+str(model.R2)+" Q^2="+str(model.Q2))    
    
# Testing Purposes
#path_to_data = 'C:\\Users\\csa97\\Research\\Projects\\DuLab\\ADAP-ML\\adap-ml\\data\\SCLC_study_output_filtered_2.csv'
#path_to_resp = 'C:\\Users\\csa97\\Research\\Projects\\DuLab\\ADAP-ML\\adap-ml\\data\\SCLC_study_responses_2.csv'
#
#data = adapml_data.DataImport(path_to_data)
#response1D = adapml_data.DataImport.getResponse(path_to_resp);
#response2D = adapml_data.DataImport.getDummyResponse(response1D);
#
#variables = data.getVariableNames()
#samples = data.getSampleNames()
#
#data.normalizeData()
#
#pca = chemo.Chemometrics(data.data, "pca", response1D)
#tr_data = pca.rotated_data
##svm = SVC(kernel='linear', C=1.0, shrinking=True).fit(data.data, np.ravel(response1D))
#svm = Classification(tr_data, response1D, 'svm', .75, kfolds=2)

#plsda = chemo.Chemometrics(data.data, "pls-da", response2D, kfolds=10)
##classify_lda = Classification(data.data, response1D, "lda", .7, kfolds=10)
#classify_lda = Classification(data.data[:,0:3], response1D, "qda", .7, kfolds=10, reg=1) # Testing is hard becuase there is not enough data
#
#y_pred_lda = classify_lda.classifier.predict(classify_lda.test_data)
#y_actu_lda = np.reshape(classify_lda.test_resp, y_pred_lda.shape)
#
#print("LDA Classification")
#print(report.classification_report(y_actu_lda, y_pred_lda))
#print("Confusion Matrix")
#print(report.confusion_matrix(np.array(y_actu_lda), np.array(y_pred_lda)))
#
#print("Actual Classes:    "+str(y_actu_lda))
#print("Predicted Classes: "+str(y_pred_lda))