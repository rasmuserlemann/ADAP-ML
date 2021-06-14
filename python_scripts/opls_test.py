# Test implementation of OPLS - Orthogonal Projection to Latent Structures
# Algorithm - Orthogonal Projections to Latent Structures (O-PLS), Wold et al. 2002
# Author - Chris Avery
# Date Initialized - February 21, 2020

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import loadTestData as load_data
import sklearn.preprocessing as pre
from sklearn.cross_decomposition import PLSRegression as PLS

class OPLS:
    def __init__(self, num_comp, resp):
        self.comp = num_comp;
        self.responses = resp
        
    def fit(self, data, resp):
        dof = data.shape[1]
        n = data.shape[0]
        ortho_comp = dof - self.comp
        
        W = np.ndarray(shape=(dof, ortho_comp))
        P = np.ndarray(shape=(dof, ortho_comp))
        T = np.ndarray(shape=(n, ortho_comp))
        
        # Start with Vector
        w = np.transpose(np.matmul(np.transpose(resp), data) / lin.norm(resp))
        
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
        
        ## Build PLS Regression from data.P
        pls = PLS(n_components=self.comp).fit(self.data_p, resp)
        self.analysis = pls
        self.rotated_data = pls.transform(self.data_p)
        
        return(self)
    
    def getVIP(self, pls_model):
        T = pls_model.x_scores_
        W = pls_model.x_weights_
        B = pls_model.x_rotations_
        
        p = pls_model.x_weights_.shape[0]
        A = pls_model.x_weights_.shape[1]
        
        vip = np.ndarray(p)
        for j in range(p):
            tmp1 = 0
            tmp2 = 0
            for a in range(A):
                tmp1 = tmp1 + W[j,a]**2 * np.linalg.norm(B[:,a],2) * np.linalg.norm(T[:,a],2)
                tmp2 = tmp2 + np.linalg.norm(B[:,a],2) * np.linalg.norm(T[:,a],2)
            
            vip[j] = tmp1/tmp2
        
        return vip
     
    def plotVIP(self, var_names):
        vip = self.vip
        n = vip.shape[1]
        
        mu = np.ndarray(n)
        sd = np.ndarray(n)
        
        for i in range(n):
            mu[i] = np.mean(vip[:,i])
            sd[i] = np.std(vip[:,i])
        
        print("VIP Scores for original features with cross validation errors")   
        print("Cross Validation errors: "+str(self.cv_error))
        plt.figure(figsize=(12, 3))
        plt.errorbar(np.arange(n), mu, yerr=sd)
        plt.xticks(np.arange(len(var_names)), var_names)
        plt.ylabel("VIP Score")
        plt.xlabel("Variables")
    
    def plotProjectionScatterMultiClass(self, num_modes, **kwargs):
        print("Projections of data into latent space.")
        print("Data is colored by response")
        pc = self.rotated_data
        if "response" in kwargs:
            resp = kwargs.get("response")
        else:
            resp = self.responses
        
        plt.figure(figsize=(24, 18))
    
        for i in range(num_modes):
            for j in range(num_modes):
                plt.subplot(5,5,5*(i) + j + 1)
                
                mylabel = list()
                
                for c in range(resp.shape[1]):
                    inx = np.where(resp[:,c] == 1)[0]
                    tmp = pc[inx,:]
                    pc1 = tmp[:,i]
                    pc2 = tmp[:,j]
                    plt.scatter(pc1, pc2)
                    mylabel.append("Class "+str(c+1))
                
                plt.xlabel("Component "+str(i+1))
                plt.ylabel("Component "+str(j+1))
                
                if "labels" in kwargs:
                    plt.legend(kwargs.get("labels"))
                else:
                    plt.legend(mylabel)
    
        plt.show()

data = load_data.loadDataPandas('../data/SCLC_study_output_filtered_2.csv')
d = data.to_numpy()
var_index = data.columns.values.tolist()

# vector of class responses associated with data
resp = load_data.getResponseMatrix1D()
resp2 = load_data.getResponseMatrix2D()

#### Create object to normalize and un-normalize data
norm_trans = pre.StandardScaler().fit(d)
data_norm = norm_trans.transform(d)

#### Train OPLS
opls = OPLS(2, resp2).fit(data_norm, resp)

#### Train PLS for comparison
pls = PLS(2).fit(data_norm, resp)
pls.rotated_data = pls.transform(data_norm)
pls.responses = resp2

#### Figures
opls.plotProjectionScatterMultiClass(2, labels=["Healthy", "Not Healthy"])
OPLS.plotProjectionScatterMultiClass(pls, 2, labels=["Healthy", "Not Healthy"])

plt.figure()
plt.plot(opls.analysis.coef_[:,0]**2)
#plt.plot(opls.analysis.coef_[:,1]**2)
plt.title("OPLS Weights")

plt.figure()
plt.plot(pls.coef_[:,0]**2)
#plt.plot(pls.x_weights_[:,1]**2)
plt.title("PLS Weights")