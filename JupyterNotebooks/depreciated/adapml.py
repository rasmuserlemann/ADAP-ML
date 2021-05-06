# Machine Learning and Statistics Library for ADAP
# Focus: Clustering
# Author: Chris Avery
# Inital Update: February 5 2020

#Basic Libraries
import numpy as np
import numpy.linalg as lin
import pandas as pd
import matplotlib.pyplot as plt
#Scipy Libraries
import scipy.cluster.hierarchy as hier
import scipy.stats as stat
#Scikit-Learn Libraries
import sklearn.preprocessing as pre
import sklearn.cluster as clst
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold

class DataImport:
    def __init__(self, path0):
        self.path = path0
        self.data = self.loadDataNumpy()
        
    def loadDataPandas(self, cols2obs=False):
        df = pd.read_csv(self.path, index_col=0)
        
        #handle case where columns are observations
        if cols2obs:
            df = df.transpose()
            
        return df
    
    def loadDataNumpy(self, cols2obs=False):
        df = self.loadDataPandas(cols2obs) #Use prior pandas function for ease
        numpy_array = df.to_numpy() #Exchange to numpy
        
        return numpy_array
    
    def getResponse(path):
        r = pd.read_csv(path, index_col=0)
        r = r.to_numpy()
        return r
    
    def getDummyResponse(resp):
        classes = np.unique(resp)
        c = len(classes)
        twod_resp = np.zeros(shape=(len(resp), c))
        
        for r in range(len(resp)):
            e = resp[r]
            twod_resp[r,e] = 1
        
        return twod_resp
    
    def getVariableNames(self, cols2obs=False):
        dfdata = self.loadDataPandas(cols2obs)
        var_names = dfdata.columns.values.tolist()
        
        return var_names
        
    def getSampleNames(self, cols2obs=False):
        dfdata = self.loadDataPandas(cols2obs)
        samp_names = dfdata.index.values.tolist()
        
        return samp_names 
    
    def normalizeData(self):
        norm_trans = pre.StandardScaler().fit(self.data)
        self.data = norm_trans.transform(self.data)
      
    def getResponseMatrix2D():
        resp1 = np.concatenate( (np.zeros([20, 1]), np.ones([20,1])), axis=0)
        resp2 = np.concatenate( (np.ones([20, 1]), np.zeros([20,1])), axis=0)
        resp = np.concatenate( (resp1, resp2), axis=1)
        
        return resp
    
    def getResponseMatrix1D():
        resp1 = np.concatenate( (np.zeros([20, 1]), np.ones([20,1])), axis=0)
        return resp1

    
        
class Clustering:
    def __init__(self, data0, method0, num_clusters0):
        self.data = data0
        self.method = method0
        self.num_clusters = num_clusters0
        
        if (self.method == "kmeans"):
            self.cluster = self.kmeans()
        elif (self.method == "hierarchical"):
            self.cluster = self.hierarchical()
        else:
            print("Clustering Method "+self.method+" Not Available")
            
    def kmeans(self):
        clust = clst.KMeans(self.num_clusters).fit(self.data)
        
        return clust
        
    def hierarchical(self):
        aff='cosine' #distance metric: {euclidean, manhattan, cosine, precomputted}
        link='average' #cluster linkage metric: {average, complete, single, ward}
        clust = clst.AgglomerativeClustering(n_clusters=self.num_clusters,
                                      affinity=aff,
                                      linkage=link)
        clust.fit(self.data)
        
        return clust
    
    ## Support Methods
    def getClusterResults(self, names):
        M = len(names)
        clusters = {}
        for i in range(self.num_clusters):
                where = [k for k in range(M) if i==self.cluster.labels_[k]]
                tmp = list()
                for k in where:
                    tmp.append(names[k])
                clusters[i] = tmp
    
        clusters_results = pd.DataFrame()
        cluster_names = {}
        for cluster in range(self.num_clusters):
            tmp = pd.DataFrame()
            tmp.insert(column=cluster, value=clusters[cluster], loc=0)
            clusters_results = pd.concat([clusters_results, tmp], ignore_index=True, axis=1)
            cluster_names[cluster] = "Cluster "+str(cluster+1)
    
        clusters_results.rename(columns=cluster_names, inplace=True)
    
        print(clusters_results)
    
    def plot_dendrogram(self, sample_labels):
        # Authors: Mathew Kallada (found online)
        # Link: https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/
        #               plot_hierarchical_clustering_dendrogram.py
        #
        # Children of hierarchical clustering
        children = self.cluster.children_
    
        # Distances between each pair of children
        # Since we don't have this information, we can use a uniform one for plotting
        distance = np.arange(children.shape[0])
    
        # The number of observations contained in each cluster level
        no_of_observations = np.arange(2, children.shape[0]+2)
    
        # Create linkage matrix and then plot the dendrogram
        linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    
        # Plot the corresponding dendrogram
        plt.figure(figsize=(14, 9))
        hier.dendrogram(linkage_matrix, labels=sample_labels)
    
class Chemometrics:
    def __init__(self, data0, method0, response0, **kwargs):
        self.data = data0
        self.method = method0
        self.responses = response0
        
        if (len(kwargs) > 0):
            if "kfolds" in kwargs:
                self.kfolds = kwargs.get("kfolds")
            if "num_comp" in kwargs:
                self.num_comp = kwargs.get("num_comp");
        else:
            self.kfolds = 1
        
        if (self.method == "pca"):
            self.analysis, self.rotated_data = self.principal_component_analysis(
                    self.data)
            self.vectors = self.analysis.components_
            
        elif (self.method == "pls-da"):
            self.analysis, self.rotated_data, self.vip, self.cv_error = self.partial_least_squares(
                    self.data, self.responses, self.kfolds)
            self.vectors = np.transpose(self.analysis.x_rotations_)
            
        elif (self.method == "opls"):
            self.analysis, self.rotated_data, self.vip, self.cv_error = self.orthogonal_pls(
                    self.data, self.responses, self.num_comp, self.kfolds)
            #self.rotated_data = self.analysis.pls.transform(self.data)
            self.responses = DataImport.getDummyResponse(self.responses)
            self.vectors = np.transpose(self.analysis.x_rotations_)
            
        elif (self.method == "lda"):
            self.analysis, self.rotated_data = self.linear_discriminant_analysis(
                    self.data, self.responses)
            self.vectors = self.analysis.coef_
            
        else:
            print("Chemometric Analysis "+self.method+" Not Found")
        
    def principal_component_analysis(self, data):
        analysis = PCA().fit(data)
        princ_comps = analysis.transform(data)
        
        return analysis, princ_comps
    
    def partial_least_squares(self, data, resp, n_cross_val):
        crossval = KFold(n_splits=n_cross_val, shuffle=True)
        cv = crossval.split(data, resp)
        p = data.shape[1]
        
        models = list()
        error = np.ndarray(n_cross_val)
        vip = np.ndarray((n_cross_val, p))
        num = 0;
        for (train, test) in cv:
            train = np.array(train); test = np.array(test);
            
            models.append(PLSRegression().fit(data[train,:], resp[train,:]))
            models.__getitem__(num).pls_comps = models.__getitem__(num).transform(data)
            
            test_pred = models.__getitem__(num).predict(data[test,:])
            error[num] = np.sqrt( np.mean( (test_pred - resp[test,:])**2 ) )
            
            vip[num, :] = self.getVIP( models.__getitem__(num) )
            
            num = num + 1
        
        best = models.__getitem__(np.argmin(error))
        analysis = best
        pls_comps = best.pls_comps
            
        return analysis, pls_comps, vip, error
    
    def orthogonal_pls(self, data, resp, num_modes, n_cross_val):
        crossval = KFold(n_splits=n_cross_val, shuffle=True)
        cv = crossval.split(data, resp)
        p = data.shape[1]
        
        opls = OPLS(num_modes).fit(data, resp)
        data_p = opls.data_p
        
        models = list()
        error = np.ndarray(n_cross_val)
        vip = np.ndarray((n_cross_val, p))
        num = 0;
        for (train, test) in cv:
            train = np.array(train); test = np.array(test);
            
            #models.append(OPLS(num_modes).fit(data[train,:], resp[train,:]))
            models.append(PLSRegression().fit(data[train,:], resp[train,:]))
            models.__getitem__(num).pls_comps = models.__getitem__(num).transform(data_p)
            
            test_pred = models.__getitem__(num).predict(data[test,:])
            error[num] = np.sqrt( np.mean( (test_pred - resp[test,:])**2 ) )
            
            vip[num, :] = self.getVIP( models.__getitem__(num) )
            
            num = num + 1
        
        best = models.__getitem__(np.argmin(error))
        analysis = best
        pls_comps = best.pls_comps
            
        return analysis, pls_comps, vip, error
        
    
    def linear_discriminant_analysis(self, data, resp):
        analysis = LinearDiscriminantAnalysis(solver='eigen', shrinkage=None)
        analysis.fit(data, np.ravel(resp))
        lda_comps = analysis.transform(data)
        
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
        
class OPLS:
    def __init__(self, num_comp):
        self.comp = num_comp;
        #self.responses = resp
        
    def fit(self, data, resp):
        dof = data.shape[1]
        n = data.shape[0]
        ortho_comp = dof - self.comp
        #ortho_comp = 0
        
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
        
        ## Build PLS Regression from data.
        tmp_resp = DataImport.getDummyResponse(resp)
        pls = PLSRegression(n_components=self.comp).fit(self.data_p, tmp_resp)
        self.pls = pls
        self.rotated_data = pls.transform(self.data_p)
        
        return(self)
        
class Statistics:
    def __init__(self, data0, method0, resp0):
        self.data = data0
        self.method = method0
        self.resp = resp0
        
        if (self.method == "ttest"):
            self.t, self.p = self.two_way_t_test()
        elif (self.method == "anova"):
            # TO-DO
            print("this will do a ANOVA test")
        
    def two_way_t_test(self):
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
        plt.legend(["p-values", ".05 threshold"])
        plt.show()
        
        