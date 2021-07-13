# Machine Learning and Statistics Library for ADAP
# Focus: Statistics
# Author: Chris Avery, Rasmus Erlemann
# Last Update: July 10 2021

#Basic Libraries
import numpy as np
import matplotlib.pyplot as plt
#Scipy Libraries
import scipy.stats as stat

class Statistics:
    def __init__(self, data0, method0, resp0):
        self.data = data0
        self.method = method0
        self.resp = resp0
        
        if (self.method == "ttest"):
            self.score, self.p = self.two_way_t_test_()
        elif (self.method == "anova"):
            self.score, self.p = self.anova_test()
        
    def two_way_t_test_(self):
        classes = np.unique(self.resp)
        c = len(classes)
        
        n_var = self.data.shape[1]
        
        t = np.zeros(shape=(n_var))
        p = np.zeros(shape=(n_var))
        for i in range(n_var):
            tmp_data = list()
            for j in range(c):
                inx = np.where(self.resp == classes[j])
                tmp_data.append(self.data[inx[0], i])
            
            t[i], p[i] = stat.ttest_ind(tmp_data[0], tmp_data[1])
        return t, p
    
    def anova_test(self):
        classes = np.unique(self.resp)
        c = len(classes)
        
        n_var = self.data.shape[1]
        
        f = np.zeros(shape=(n_var))
        p = np.zeros(shape=(n_var))
        for i in range(n_var):
            tmp_data = list()
            for j in range(c):
                inx = np.where(self.resp == classes[j])
                tmp_data.append(self.data[inx[0], i])
            
            f[i], p[i] = stat.f_oneway(*tmp_data)
            
        return f, p
    
    ## Support methods
    def plot_logp_values(self, variables):
        p = self.p
        logp = -1*np.log10(p)
        
        thresh = -1*np.log10(.05)
        
        var = np.arange(self.data.shape[1])
        
        plt.figure(figsize=(10, 5))
        plt.scatter(var, logp)
        
        plt.xticks(var, variables)
        plt.plot(thresh*np.ones(shape=(len(var),1)), c='r')
        plt.legend([".05 threshold", "p-values"])
        plt.ylabel("-log10(p-value)")
        plt.title("P-Value Significance")
        plt.show()
        
    def plot_volcano_t(self, variables):
        classes = np.unique(self.resp)
        c = len(classes)
        
        p = -1*np.log10(self.p)
        
        mu = np.zeros(shape=(len(p),c))
        for i in range(len(p)):
            for j in range(c):
                inx = np.where(self.resp == classes[j])
                mu[i,j] = np.mean(self.data[inx, i])
                
        fc = -1*np.log10(mu[:,0]/mu[:,1])
        
        plt.figure(figsize=(10,5))
        plt.scatter(fc, p)
        plt.xlabel("-log10(FC)")
        plt.ylabel("-log10(p)")
        plt.title("Volcano Plot")
        for i in range(len(variables)):
            plt.annotate(variables[i], xy=(fc[i],p[i]),
                         xytext=(5,5), textcoords="offset points")
        
        plt.show()


"""
import adapml_data
import os
##### TESTING CODE 1
reldir = os.getcwd()
#path_to_resp = os.path.join(reldir, '..', 'data', 'SCLC_study_responses_2.csv')
path_to_data = os.path.join(reldir, '..', 'data', 'SCLC_study_output_filtered_2.csv')

data = adapml_data.DataImport(path_to_data)
response1D = adapml_data.DataImport.getResponse(path_to_data)
#resp1 = data.getResponseNew
#response2D = adapml_data.DataImport.getDummyResponse(response1D);


variables = data.getVariableNames()
samples = data.getSampleNames()
tmodel = Statistics(data.data, 'anova', response1D)
tmodel.plot_logp_values(variables)
tmodel.plot_volcano_t(variables)
"""

##### TESTING CODE 2    
#import adapml_data
#path_to_data = 'C:\\Users\\csa97\\Research\\Projects\\DuLab\\ADAP-ML\\adap-ml\\data\\SCLC_study_output_filtered_2.csv'
#path_to_resp = 'C:\\Users\\csa97\\Research\\Projects\\DuLab\\ADAP-ML\\adap-ml\\data\\SCLC_study_responses_2.csv'
#data = adapml_data.DataImport(path_to_data)
#response1D = adapml_data.DataImport.getResponse(path_to_resp);
#response2D = adapml_data.DataImport.getDummyResponse(response1D);
#variables = data.getVariableNames()
#samples = data.getSampleNames()
##data.normalizeData()
#
#tmodel = Statistics(data.data, 'anova', response1D)
#tmodel.plot_logp_values(variables)
#tmodel.plot_volcano_t(variables)