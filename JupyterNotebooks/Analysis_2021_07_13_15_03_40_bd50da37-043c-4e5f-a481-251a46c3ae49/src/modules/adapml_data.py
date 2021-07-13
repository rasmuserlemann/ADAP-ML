# Machine Learning and Statistics Library for ADAP
# Focus: Data Import
# Author: Chris Avery
# Last Update: February 24 2020

#Basic Libraries
import numpy as np
import pandas as pd
#Scikit-Learn Libraries
import sklearn.preprocessing as pre

class DataImport:
    def __init__(self, path0):
        self.path = path0
        self.data = self.loadDataNumpy()
        self.resp = self.getResponseNew()
        
    def loadDataPandas(self, cols2obs=False):
        df = pd.read_csv(self.path, index_col=0)
        
        #handle case where columns are observations
        if cols2obs:
            df = df.transpose()
            
        return df
    
    def loadDataNumpy(self, cols2obs=False):
        df = self.loadDataPandas(cols2obs) #Use prior pandas function for ease
        numpy_array = df.to_numpy() #Exchange to numpy
        
        return numpy_array[:,1:]
    
    def getResponse(path):
        r = pd.read_csv(path, index_col=0)
        r = pd.DataFrame(r["Response"]).to_numpy()
        #resp = r[:,0].astype(int)
        #resparray = pd.DataFrame([[el] for el in resp])
        return r

    def getResponseNew(self):
        resphalf = int(self.data.shape[0]/2)
        r = [0 for x in range(resphalf)]
        for x in range(resphalf): r.append(1)
        r = pd.DataFrame(r)
        r = r.to_numpy()
        return r
    
    def getDummyResponse(resp):
        classes = np.unique(resp)
        c = len(classes)
        twod_resp = np.zeros(shape=(len(resp), c))
        
        clas = list(classes)
        for r in range(len(resp)):
            e = clas.index(resp[r])
            twod_resp[r,e] = 1
        
        return twod_resp
    
    
    def flattenDummyResp(resp):
        m = resp.shape[0]
        flat = np.zeros(shape=(m,1))
        
        for i in range(m):
            flat[i] = np.where(resp[i,:]==1)
            
        return flat
    
    def getVariableNames(self, cols2obs=False):
        dfdata = self.loadDataPandas(cols2obs)
        var_names = dfdata.columns.values.tolist()
        
        return var_names[1:]
        
    def getSampleNames(self, cols2obs=False):
        dfdata = self.loadDataPandas(cols2obs)
        samp_names = dfdata.index.values.tolist()
        
        return samp_names 
    
    def normalizeData(self, method):
        if( method == "autoscale"):
            norm_trans = pre.StandardScaler().fit(self.data)
            self.data = norm_trans.transform(self.data)
        elif( method == "meancenter"):
            norm_trans = pre.scale(self.data, with_mean='True', with_std='False')
            #self.data = norm_trans.transform(self.data)
        elif( method == "minmax"):
            norm_trans = pre.MinMaxScaler().fit(self.data)
            self.data = norm_trans.transform(self.data)
        else:
            print("Normalization method not recognized, Proceeding without normalizing!")
      
    def getResponseMatrix2D():
        resp1 = np.concatenate( (np.zeros([20, 1]), np.ones([20,1])), axis=0)
        resp2 = np.concatenate( (np.ones([20, 1]), np.zeros([20,1])), axis=0)
        resp = np.concatenate( (resp1, resp2), axis=1)
        
        return resp
    
    def getResponseMatrix1D():
        resp1 = np.concatenate( (np.zeros([20, 1]), np.ones([20,1])), axis=0)
        return resp1
