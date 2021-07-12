# Machine Learning and Regression Library for ADAP
# Focus: Regression
# Author: Rasmus Erlemann
# Last Update: July 9 2021

#Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#skikit Libraries
from sklearn.linear_model import LinearRegression

class Regression:
    def __init__(self, data0, method0):
        self.data = data0
        self.method = method0

        if (self.method == "linregression"):
            self.regression = self.linregression_()
        #elif (self.method == "hierarchical"):
            #self.cluster = self.hierarchical_()
        else:
            print("Regression method "+self.method+" Not Available")

    def linregression_(self):
        reg = LinearRegression().fit(self.data)

        return reg.score(self.data)

import adapml_data
import os
###Test code
reldir = os.getcwd()
path_to_resp = os.path.join(reldir, '..', 'data', 'SCLC_study_responses_2.csv')
path_to_data = os.path.join(reldir, '..', 'data', 'SCLC_study_output_filtered_2.csv')

data = adapml_data.DataImport(path_to_data)
print(data.data)

linregression = Regression(data.data, "linregression")
print(linregression.linregression())

