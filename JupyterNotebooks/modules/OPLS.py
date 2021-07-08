# Machine Learning and Statistics Library for ADAP
# Focus: Clustering
# Author: Chris Avery
# Last Update: February 24 2020

#Basic Libraries
import numpy as np
import numpy.linalg as lin
#Scikit-Learn Libraries
from sklearn.cross_decomposition import PLSRegression
#Local Libraries
import modules.adapml_data as adapml_data
        
class opls:
    def __init__(self, num_comp):
        self.comp = num_comp;
        #self.responses = resp
        
    def fit(self, data, resp):
        dof = data.shape[1]
        n = data.shape[0]
        ortho_comp = dof - self.comp
        #ortho_comp = 0
        
        # print(dof)
        # print(ortho_comp)
        
        W = np.ndarray(shape=(dof, ortho_comp))
        P = np.ndarray(shape=(dof, ortho_comp))
        T = np.ndarray(shape=(n, ortho_comp))
        
        # Start with Vector
        w = np.transpose(np.matmul(np.transpose(resp), data))
        w = w/lin.norm(resp)
        
        for i in range(ortho_comp):
            t = np.matmul(data, w) / np.matmul(np.transpose(w), w) # get pls scores
            p = np.transpose(np.matmul(np.transpose(t), data) / np.matmul(np.transpose(t), t)) # pls loadings
            
            ## Get Orthogonal Components
            w_ortho = p - ((np.matmul(np.transpose(w), p) / np.matmul(np.transpose(w), w)) * w)
            w_ortho = w_ortho / lin.norm(w_ortho)
            t_ortho = np.matmul(data, w_ortho) / np.matmul(np.transpose(w_ortho), w_ortho)
            p_ortho = np.transpose(np.matmul(np.transpose(t_ortho), data) / np.matmul(np.transpose(t_ortho), t_ortho) )
            
            data = data - np.matmul(t_ortho, np.transpose(p_ortho))
            W[:,i] = np.reshape(w_ortho, (dof,));
            P[:,i] = np.reshape(p_ortho, (dof,));
            T[:,i] = np.reshape(t_ortho, (n,))  ;
        
        self.data_p = data
        self.data_o = np.matmul(T, np.transpose(P))
        self.W_o = W
        self.T_o = T
        self.P_o = P
        
        ## Build PLS Regression from data.
        tmp_resp = adapml_data.DataImport.getDummyResponse(resp)
        pls = PLSRegression(n_components=self.comp).fit(self.data_p, tmp_resp)
        self.pls = pls
        self.rotated_data = pls.transform(self.data_p)
        
        return(self)
        