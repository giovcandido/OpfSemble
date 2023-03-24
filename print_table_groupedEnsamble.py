import numpy as np
from scipy.stats import wilcoxon
import os



class Algorithms(object):
    def __init__(self):
        self.algorithms_opf = ['OPF_Ensemble_mode','OPF_Ensemble_average','OPF_Ensemble_intercluster','OPF_Ensemble_mode_best']
        self.alias_algorithms_opf = ['Mode','Average','Intercluster','Mode Best']

        self.algorithms_superLearner = ['Super_Learner','Super_Learner_Full']
        self.alias_algorithms_superLearner = ['SuperLearner','SuperLearner Full']

        self.algorithms_divPruning = ['MoCART','MoKNN','MoMLP','MoNB','MoRCART','MoRKNN','MoRMLP','MoRNB','PrCART','PrKNN','PrMLP','PrNB']
        self.alias_algorithms_divPruning = ['MoCART','MoKNN','MoMLP','MoNB','MoRCART','MoRKNN','MoRMLP','MoRNB','PrCART','PrKNN','PrMLP','PrNB']

        self.algorithms_subspace = ['AggrCART','AggrKNN','AggrMLP','AggrNB','MVCART','MVKNN','MVMLP','MVNB']
        self.alias_algorithms_subspace = ['AggrCART','AggrKNN','AggrMLP','AggrNB','MVCART','MVKNN','MVMLP','MVNB']
        
        
        self.algorithms_baseline = ['OPF','KNN', 'Extra Trees','Gradient Boosting','MLP','Naive Bayes','Random Forest','SVM']
        self.alias_algorithms_baseline = ['OPF','KNN', 'Extra Trees','Gradient Boosting','MLP','Naive Bayes','Random Forest','SVM'] 
        
        
        
    def getAlgorithm(self, tipo):
        if tipo=='opf':
            return self.algorithms_opf
        elif tipo=='superLearner':
            return self.algorithms_superLearner
        elif tipo=='divPruning':
            return self.algorithms_divPruning
        elif tipo=='subspace':
            return self.algorithms_subspace
        elif tipo=='baseline':
            return self.algorithms_baseline
        
    def getAliasAlgorithm(self, tipo):
        if tipo=='opf':
            return  self.alias_algorithms_opf
        elif tipo=='superLearner':
            return  self.alias_algorithms_superLearner
        elif tipo=='divPruning':
            return  self.alias_algorithms_divPruning
        elif tipo=='subspace':
            return  self.alias_algorithms_subspace
        elif tipo=='baseline':
            return  self.alias_algorithms_baseline
        
    def getAliasAndAlgorithm(self, tipo):
        if tipo=='opf':
            return self.algorithms_opf,  self.alias_algorithms_opf
        elif tipo=='superLearner':
            return self.algorithms_superLearner,  self.alias_algorithms_superLearner
        elif tipo=='divPruning':
            return self.algorithms_divPruning,  self.alias_algorithms_divPruning
        elif tipo=='subspace':
            return self.algorithms_subspace,  self.alias_algorithms_subspace
        elif tipo=='baseline':
            return self.algorithms_baseline,  self.alias_algorithms_baseline
            
            
            
    def getLen(self, tipo):        
        return len(self.getAlgorithm(tipo))

    def getAlgorithmConcat(self, tipo):        
        alg_opf, alias_opf = self.getAliasAndAlgorithm('opf')
        alg_comp, alias_alg_comp = self.getAliasAndAlgorithm(tipo)
        
        
        try:
            return  np.concatenate((alg_opf,alg_comp), axis=0), np.concatenate((alias_opf,alias_alg_comp), axis=0)
        except:
            return  alg_opf, alias_opf


class PT(object):
    def __init__(self, datasets,algorithms,n_classifiers,resultsFolder, tipo):
        self.datasets=datasets
        self.n_classifiers=n_classifiers
        self.resultsFolder=resultsFolder
        self.tipo=tipo
        self.algorithms=algorithms
        self.alias_algorithms_subspace_temp = None
        

    def __load(self, ds, n_clas, alg):
        X = []
        for i in np.arange(1,21):
            if self.tipo =='divPruning':
                temp = np.loadtxt('{}/{}/{}/{}/{}/{}/results.txt'.format(self.resultsFolder,self.tipo,self.algorithms[alg],self.datasets[ds],str(i),self.n_classifiers[n_clas]),delimiter=',', dtype='str')
                temp=temp[temp[:,0]==self.algorithms[alg]]
                print(temp)
                
                
            else:
                X.append(np.loadtxt('{}/{}/{}/{}/{}/{}/results.txt'.format(self.resultsFolder,self.tipo,self.algorithms[alg],self.datasets[ds],str(i),self.n_classifiers[n_clas]),delimiter=',', max_rows=1,usecols=(1,2,3,4)))	    

        X = np.asarray(np.vstack(X))
        return X

    def calcularValores(self, ds,n_clas):
        # # algorithms, # statistics
        mat = np.zeros((len(self.algorithms),8))

        for alg in range(len(self.algorithms)):              
            X = self.__load(ds,n_clas,alg)

            mat[alg,0] = np.average(X[:,0])#best k
            mat[alg,1] = np.std(X[:,0])#std Best k
            mat[alg,2] = np.average(X[:,1])#accuracy
            mat[alg,3] = np.std(X[:,1])#std accuracy
            mat[alg,4] = np.average(X[:,2])#f1
            mat[alg,5] = np.std(X[:,2])#std f1
            mat[alg,6] = np.average(X[:,3])#time
            mat[alg,7] = np.std(X[:,3])#std time

        return mat

    def calcularWilcoxon(self, ds,n_clas, mat, index_metric=2):
        #index_metric: 0 = k/k_max, 1 = Accuracy, 2 = F1
        # # algorithms, # metaheuristic tech
        clAlgo = Algorithms()
        self.algorithms = np.concatenate((np.concatenate((clAlgo.algorithms_opf,clAlgo.algorithms_divPruning), axis=0),clAlgo.algorithms_subspace), axis=0)

                
        wil = np.zeros(len(clAlgo.algorithms_opf)+  len(clAlgo.algorithms_divPruning)+  len(clAlgo.algorithms_subspace))
        alg_best= mat.argmax()
        
        
        if alg_best<clAlgo.getLen('opf'):
            self.tipo = 'opf'
        elif alg_best<clAlgo.getLen('opf')+clAlgo.getLen('divPruning'):
            self.tipo = 'divPruning'
        else:
            self.tipo = 'subspace'

        better =self.__load(ds,n_clas, alg_best)
        
        better = better[:,index_metric]
        
        better_avg = np.average(better)
        wil[alg_best] = 2
        for alg in range(len(self.algorithms)):
            if alg_best !=alg:            
            
                if alg<clAlgo.getLen('opf'):
                    self.tipo = 'opf'
                elif alg<clAlgo.getLen('opf')+clAlgo.getLen('divPruning'):
                    self.tipo = 'divPruning'
                else:
                    self.tipo = 'subspace'

                x =self.__load(ds,n_clas, alg)       

                if better_avg==np.average(x[:,index_metric]):
                    wil[alg] = 2
                else:
                    statistic, pvalue = wilcoxon(better,x[:,index_metric])
                    if pvalue>=0.05:
                        wil[alg] = 1

        return wil
