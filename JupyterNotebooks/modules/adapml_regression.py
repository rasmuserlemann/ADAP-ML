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
    def __init__(self, data0, method0):
        self.data = data0
        self.method = method0

        if (self.method == "linear"):
            self.linear = self.linear_()
        #elif (self.method == "hierarchical"):
            #self.cluster = self.hierarchical_()
        else:
            print("Regression method "+self.method+" Not Available")
    def linear_(self):
        nrvar = int(self.data.shape[1])
        R2mat = np.zeros((nrvar,nrvar))
        for var1 in range(nrvar):
            for var2 in range(nrvar):
                x = self.data[:,var1]
                y = self.data[:,var2]
                R2mat[var1,var2] = max(0,r2_score(x,y))
        plt.imshow(R2mat, cmap='hot', interpolation='nearest')
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)
        plt.show()

"""
import adapml_data
import os
###Test code
reldir = os.getcwd()
path_to_data = os.path.join(reldir, '..', 'data', 'SCLC_study_output_filtered_2.csv')

data = adapml_data.DataImport(path_to_data)
print(data.data)

test = Regression(data.data, "linear")
"""
