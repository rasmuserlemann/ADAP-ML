# Machine Learning and Regression Library for ADAP
# Focus: Regression
# Author: Rasmus Erlemann
# Last Update: July 9 2021

#Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#skikit Libraries
from sklearn.metrics import r2_score

class Regression:
    def __init__(self, data0, method0, R2lim0):
        self.data = data0
        self.method = method0
        self.R2lim = R2lim0

        if (self.method == "linear"):
            self.linear = self.linear_()
        #elif (self.method == "hierarchical"):
            #self.cluster = self.hierarchical_()
        else:
            print("Regression method "+self.method+" Not Available")
    def linear_(self):
        nrvar = int(self.data.shape[1])
        self.R2mat = np.zeros((nrvar,nrvar))
        xvar = []
        yvar = []
        for var1 in range(nrvar):
            for var2 in range(nrvar):
                x = self.data[:,var1]
                y = self.data[:,var2]
                score = max(0,r2_score(x,y))
                self.R2mat[var2,var1] = score
                if score > self.R2lim and var1!=var2:
                    xvar.append(var1)
                    yvar.append(var2)
        plt.imshow(self.R2mat, cmap='hot', interpolation='nearest')
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)
        plt.show()
        self.indices = [xvar,yvar]

    def DisplaySampleNames(self, labels):
        var1 =  self.indices[0]
        var2 = self.indices[1]
        for var in range(len(self.indices[0])):
            print("R2 score between " + labels[var1[var]] + " and " + labels[var2[var]] + " is " + str(self.R2mat[var2[var],var1[var]]))
'''
import os
os.chdir('/Users/rerleman/Documents/Git/adap-ml/JupyterNotebooks/modules')

import adapml_data
###Test code
reldir = os.getcwd()
path_to_data = os.path.join(reldir, '..', 'data', 'SCLC_study_output_filtered_2.csv')

data = adapml_data.DataImport(path_to_data)
print(data.data)

test = Regression(data.data, "linear", 0.25)
test.DisplaySampleNames(data.getSampleNames())
'''
