# Machine Learning and Statistics Library for ADAP
# Focus: Chemometrics
# Author: Chris Avery
# Last Update: February 24 2020

#Basic Libraries
import numpy as np
import matplotlib.pyplot as plt
#Scikit-Learn Libraries
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr 
#Local Libraries
import adapml_data
import adapml_classification
import OPLS

    
class Chemometrics:
    def __init__(self, data0, method0, response0, **kwargs):
        self.data = data0
        self.method = method0
        self.responses = response0
        self.train_data, self.test_data, self.train_resp, self.test_resp = train_test_split(
                self.data, self.responses, test_size=.25)
        
        if (len(kwargs) > 0):
            if "kfolds" in kwargs:
                self.kfolds = kwargs.get("kfolds");
                
            if "num_comp" in kwargs:
                self.num_comp = kwargs.get("num_comp");
            else:
                self.num_comp = 2;
                
            if "opls_comp" in kwargs:
                self.opls_comp = kwargs.get("opls_comp"); 
            
        else:
            self.kfolds = 1
        
        if (self.method == "pca"):
            self.analysis, self.rotated_data = self.principal_component_analysis_(
                    self.data)
            self.vectors = self.analysis.components_
            self.responses = adapml_data.DataImport.getDummyResponse(self.responses)
            
        elif (self.method == "pls-da"):
            tmp_resp = adapml_data.DataImport.getDummyResponse(self.train_resp)
            self.analysis, self.rotated_data, self.vip, self.cv_error = self.partial_least_squares_(
                    self.train_data, tmp_resp, self.kfolds, self.num_comp)
            self.vectors = np.transpose(self.analysis.x_weights_)
            self.responses = adapml_data.DataImport.getDummyResponse(self.train_resp)
            
        elif (self.method == "opls"):
            tmp_resp = adapml_data.DataImport.getDummyResponse(self.train_resp)
            self.analysis, self.rotated_data, self.vip, self.cv_error = self.orthogonal_pls_(
                    self.train_data, tmp_resp, self.num_comp, self.opls_comp , self.kfolds)
            self.responses = adapml_data.DataImport.getDummyResponse(self.responses)
            self.vectors = np.transpose(self.analysis.x_weights_)
            self.responses = adapml_data.DataImport.getDummyResponse(self.train_resp)
            
        elif (self.method == "lda"):
            self.analysis, self.rotated_data = self.linear_discriminant_analysis_(
                    self.train_data, self.train_resp)
            self.vectors = self.analysis.coef_
            self.responses = adapml_data.DataImport.getDummyResponse(self.train_resp)
            
            
        else:
            print("Chemometric Analysis "+self.method+" Not Found")
        
    def principal_component_analysis_(self, data):
        analysis = PCA().fit(data)
        princ_comps = analysis.transform(data)
        
        return analysis, princ_comps
    
    def partial_least_squares_(self, data, resp, n_cross_val, num_comp):
        crossval = KFold(n_splits=n_cross_val, shuffle=True)
        cv = crossval.split(data, resp)
        p = data.shape[1]
        
        models = list()
        error = np.ndarray(n_cross_val)
        #vip = np.ndarray((n_cross_val, p))
        vip = list()
        num = 0;
        for (train, test) in cv:
            train = np.array(train); test = np.array(test);
            
            models.append(PLSRegression(n_components=num_comp).fit(data[train,:], resp[train,:]))
            models.__getitem__(num).pls_comps = models.__getitem__(num).transform(data[train,:])
            
            test_pred = models.__getitem__(num).predict(data[test,:])
            error[num] = np.sqrt( np.mean( (test_pred - resp[test,:])**2 ) )
            
            tmp_vip = self.getVIP_( models.__getitem__(num), data[train, :], resp[train,:])
            vip.append(tmp_vip)
            
            num = num + 1
        
        best = models.__getitem__(np.argmin(error))
        analysis = best
        pls_comps = best.transform(data)
        #vip = vip[np.argmin(error), :]
        vip = vip.__getitem__(np.argmin(error))
        
        y_pred_tr = analysis.predict(self.train_data)
        y_pred = analysis.predict(self.test_data)
        self.R2 = r2_score(adapml_data.DataImport.getDummyResponse(self.train_resp), y_pred_tr)
        self.Q2 = r2_score(adapml_data.DataImport.getDummyResponse(self.test_resp), y_pred)
            
        return analysis, pls_comps, vip, error
    
    def orthogonal_pls_(self, data, resp, num_modes, opls_comp, n_cross_val):
        crossval = KFold(n_splits=n_cross_val, shuffle=True)
        cv = crossval.split(data, resp)
        p = data.shape[1]
        
        opls = OPLS.opls(opls_comp).fit(data, self.train_resp)
        data_p = opls.data_p
        
        models = list()
        error = np.ndarray(n_cross_val)
        #vip = np.ndarray((n_cross_val, p))
        vip = list()
        num = 0;
        for (train, test) in cv:
            train = np.array(train); test = np.array(test);
            
            models.append(PLSRegression(num_modes).fit(data_p[train,:], resp[train,:]))
            models.__getitem__(num).pls_comps = models.__getitem__(num).transform(data_p)
            
            test_pred = models.__getitem__(num).predict(data_p[test,:])
            error[num] = np.sqrt( np.mean( (test_pred - resp[test,:])**2 ) )
            
            tmp_vip = self.getVIP_( models.__getitem__(num), data[train, :], resp[train,:])
            vip.append(tmp_vip)
            
            num = num + 1
        
        best = models.__getitem__(np.argmin(error))
        analysis = best
        pls_comps = best.transform(data_p)
        #vip = vip[np.argmin(error),:]
        vip = vip.__getitem__(np.argmin(error))
        
        y_pred_tr = analysis.predict(data_p)
        y_pred = analysis.predict(self.test_data)
        self.R2 = r2_score(adapml_data.DataImport.getDummyResponse(self.train_resp), y_pred_tr)
        self.Q2 = r2_score(adapml_data.DataImport.getDummyResponse(self.test_resp), y_pred)
            
        return analysis, pls_comps, vip, error
        
    
    def linear_discriminant_analysis_(self, data, resp):
        analysis = LinearDiscriminantAnalysis(solver='eigen', shrinkage=None)
        analysis.fit(data, np.ravel(resp))
        lda_comps = analysis.transform(data)
        
        y_pred_tr = analysis.predict(self.train_data)
        y_pred = analysis.predict(self.test_data)
        self.R2 = r2_score(self.train_resp, y_pred_tr)
        self.Q2 = r2_score(self.test_resp, y_pred)
        
        return analysis, lda_comps
    
    ## Support Methods
    def getCumVar(self, N):
        scree = self.analysis.explained_variance_
        #N = len(scree)
        cumvar = np.empty([N]);
        for i in range(N):
            sum = 0
            for j in range(i):
                sum = sum + scree[j]
            cumvar[i] = sum
    
        return cumvar
    
    def plotScree(self, **kwargs):
        scree = self.analysis.explained_variance_
        
        if (len(kwargs) > 0):
            num_modes = kwargs.get("num_modes")
        else:
            num_modes = len(scree)
        
        sumvar = np.sum(scree)
        scree = scree[:num_modes]
        cumvar = self.getCumVar(num_modes)/sumvar
        dim = np.arange(1,len(scree)+1)
    
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_ylabel("Scree")
        ax1.plot(dim, scree, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
    
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel("% Cumulative Variance")
        ax2.plot(dim, cumvar, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
    
        plt.title("Scree Plot")
        plt.xlabel("Dimension Index")
        plt.xticks(dim)
        plt.show()
        return
    
    def plotProjectionScatter(self, num_modes):
        print("Projections of data into latent space.")
        pc = self.rotated_data
        plt.figure(figsize=(24, 18))
        for i in range(num_modes):
            mode_1 = i+1
            for j in range(5):
                mode_2 = j+1
                plt.subplot(5,5,5*(i) + j + 1)
                pc1 = pc[:,mode_1-1]
                pc2 = pc[:,mode_2-1]
                plt.scatter(pc1, pc2)
                plt.xlabel("Component "+str(mode_1))
                plt.ylabel("Component "+str(mode_2))
        plt.show()
    
    def plotProjectionScatterMultiClass(self, num_modes, **kwargs):
        print("Projections of data into latent space.")
        print("Data is colored by response")
        if "data" in kwargs:
            pc = kwargs.get("data")
        else:
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
    
    def plotVectorLoadings(self, var_names, num_vectors):
        print("Plotting the squared loadings of the latent space transformation vectors")
        print("A Larger magnitude indicates larger importance for corresponding feature")
        vectors = self.vectors
        plt.figure(figsize=(12, 3*num_vectors)) #Adaptable size to number of vecs
        for i in range(num_vectors):
            plt.subplot(num_vectors,1,i+1)
            vec = np.power(vectors[i,:], 2) # Note the eigenvectors are the row space
            plt.plot(vec)
            plt.ylabel("Squared Mode "+str(i+1))
            plt.xticks(np.arange(len(var_names)), var_names)
        plt.xlabel("Variables")
        plt.show()
    
    def plotLoadingsScatter(self, num_vectors):
        vectors = self.vectors
        plt.figure(figsize=(24, 18))
        for i in range(num_vectors):
            mode_1 = i+1
            for j in range(5):
                mode_2 = j+1
                plt.subplot(5,5,5*(i) + j + 1)
                pc1 = vectors[mode_1-1,:]
                pc2 = vectors[mode_2-1,:]
                plt.scatter(pc1, pc2)
                plt.xlabel("PC "+str(mode_1))
                plt.ylabel("PC "+str(mode_2))
        plt.show()
        
    def getVIP_(self, pls_model, data, resp):
        P = pls_model.x_loadings_
        T = pls_model.transform(data)
        Y = adapml_data.DataImport.flattenDummyResp(resp)
        
        p = P.shape[0]
        n_comp = P.shape[1]
        
        r = np.abs(corr2(T,Y))
                
        vip = np.zeros(shape=(p, n_comp))
        vip[:,0] = P[:,0]**2
        
        if(n_comp > 1):
            for i in range(1,n_comp):
                print(i)
                R = r[0:i,:]
                
                mul1 = np.transpose(R);
                mul2 = np.transpose(P[:,0:i]**2);
                vip[:,i] = np.matmul(mul1, mul2)/np.sum(R)
        
        vip = np.sqrt(vip*p)
        
        return vip
    
    def plotVIP(self, var_names, comp_num):
        vip = self.vip[:,comp_num-1]
        
        print("VIP Scores for original features with cross validation errors")   
        print("Average Cross Validation Error: "+str(np.mean(self.cv_error)))
        plt.figure(figsize=(12, 3))
        #plt.errorbar(np.arange(n), mu, yerr=sd)
        inx = np.argsort(-1*vip)
        
        var_names = [var_names[i] for i in inx]
        print()
        
        plt.plot(vip[inx])
        plt.xticks(range(vip.size-1), var_names)
        
def corr2(A, B):
    # A is the scores T
    # B is the responses Y
    ncomp = A.shape[1]
    m = A.shape[0]
    nresp = B.shape[1]
    
    a_mu = np.mean(A, axis=0); b_mu = np.mean(B, axis=0)
    a_sd = np.std(A, axis=0); b_sd = np.std(B, axis=0)
    
    r = np.zeros(shape=(ncomp, nresp))
    for i in range(ncomp):
        a = (A[:,i] - a_mu[i])/a_sd[i]
        for j in range(nresp):
            b = (B[:,j] - b_mu[j])/b_sd[j]
            
            tmp = np.matmul(np.transpose(a), b)
            r[i,j] = tmp/(m-1)
        
    return r
        