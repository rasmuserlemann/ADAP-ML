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
            self.score, self.p = self.anova_test_()
            
        self.Bon1 = self.Bonferroni()[0]
        self.Bon2 = self.Bonferroni()[1]
        self.BH1 = self.BH()[0]
        self.BH2 = self.BH()[1]
        
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
        
    def ttest_power(self):
        classes = np.unique(self.resp)
        c = len(classes)
        
        n_var = self.data.shape[1]
        
        t = np.zeros(shape=(n_var))
        p = np.zeros(shape=(n_var))
        Agroup = list()
        Bgroup = list()
        for i in range(n_var):
            tmp_data = list()
            for j in range(c):
                inx = np.where(self.resp == classes[j])
                tmp_data.append(self.data[inx[0], i])
            Agroup.append(tmp_data[0])
            Bgroup.append(tmp_data[1])
            
        Agroup = np.array(Agroup) #Array of the samples with response 0
        Bgroup = np.array(Bgroup) #Array of the samples with response 1
        
        samplesizes = range(5, Agroup.shape[0], 3) #Choose the sample sizes for the study
        nr_clusters = 4 #nr of alternatives
        sig_lev = 0.05 #significance level
        power_sim = 1000 #number of simulations to calculate the power
        control_typeII = 0.6 #Minimum power level. 1-control_typeII is the desired type II error
        
        Aclust = clst.KMeans(nr_clusters).fit(Agroup).cluster_centers_ #Cluster centers for the response 0 samples
        Bclust = clst.KMeans(nr_clusters).fit(Bgroup).cluster_centers_ #Cluster centers for the response 1 samples
        
        min_n = Agroup.shape[0] #Maximum sample size returned by the power study is the full sample size of response 0 (response 1 has the same number of samples)
        
        for n in samplesizes:
            minpow = 1 
            for alt in range(nr_clusters):
                nr_rej = 0 
                for sim in range(power_sim):
                    sample1 = Aclust[alt][0:n] 
                    sample2 = Bclust[alt][0:n] 
                    t, p = stat.ttest_ind(sample1, sample2) 
                    if p <= sig_lev:
                        nr_rej += 1
                pow = nr_rej/power_sim
                if pow <= minpow:
                    minpow = pow
            if minpow >= control_typeII:
                min_n = n
            return(min_n)
        
    
    def anova_test_(self):
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
    def Bonferroni(self):
        classes = np.unique(self.resp)
        c = len(classes)
        k = len(self.data)/c
        alpha1 = 0.05/k
        alpha2 = 0.01/k
        return([alpha1,alpha2])
    def BH(self):
        classes = np.unique(self.resp)
        c = len(classes)
        m = int(len(self.data)/c)-2
        pthres1 = 0
        pthres2 = 0
        for i in range(m):
            if sorted(self.p)[i] <= (i/m)*0.05:
                pthres1 = sorted(self.p)[i]
            if sorted(self.p)[i] <= (i/m)*0.01:
                pthres2 = sorted(self.p)[i]
        return([pthres1, pthres2])

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
tmodel = Statistics(data.data, 'ttest', response1D)
print(tmodel.ttest_power())

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