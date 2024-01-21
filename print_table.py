import numpy as np
from scipy.stats import wilcoxon
import os

class PT(object):
    def __init__(self, datasets,algorithms_opf,algorithms_comparison,n_classifiers,resultsFolder, tipo, individual=False):
        self.datasets=datasets
        self.algorithms_opf=algorithms_opf 
        self.algorithms_comparison=algorithms_comparison 
        self.n_classifiers=n_classifiers
        self.resultsFolder=resultsFolder
        self.tipo=tipo
        self.individual=individual
        self._algorithms=None
        
    def setAlgorithms(self):
        if self.algorithms_comparison is None:
            self._algorithms = self.algorithms_opf
        elif self._algorithms is None:
            self._algorithms = np.concatenate((self.algorithms_opf,self.algorithms_comparison), axis=0)
        elif not self.n_classifiers in self.algorithms_comparison:
            self._algorithms = np.concatenate((self.algorithms_opf,self.algorithms_comparison), axis=0)
        return self._algorithms

    def __load(self, ds, n_clas, alg):
        X = []
        algorithms = self.setAlgorithms()
        
        tipo = 'opf' if alg<len(self.algorithms_opf) else self.tipo
        for i in np.arange(1,21):
            if tipo == 'baseline':
                value = np.loadtxt('{}/{}/{}/{}/{}/results.txt'.format(self.resultsFolder,tipo,self.datasets[ds],str(i),self.n_classifiers[n_clas]),delimiter=',', dtype='str') 
                if self.individual:         
                    X.append(value[value[:,0]==algorithms[alg], 1:].astype(float))	
                else:
                    sub = []
                    for i in range(len(value[:,0])):
                        s = value[i,0]
                        if s.startswith(algorithms[alg]):
                            sub.append(value[i,1:].astype(float))
                    sub = np.asarray(np.vstack(sub))
                    X.append(np.mean(sub, axis=0))
            else:
                X.append(np.loadtxt('{}/{}/{}/{}/{}/{}/results.txt'.format(self.resultsFolder,tipo,algorithms[alg],self.datasets[ds],str(i),self.n_classifiers[n_clas]),delimiter=',', max_rows=1,usecols=(1,2,3,4)))	    

        X = np.asarray(np.vstack(X))
        return X
        
    def loadBaselineNames(self, ds, n_clas):
        if self.individual:
            self.algorithms_comparison = np.loadtxt('{}/{}/{}/{}/{}/results.txt'.format(self.resultsFolder,self.tipo,ds,1,self.n_classifiers[n_clas]),dtype='str',delimiter=',',usecols=(0))            
        return self.algorithms_comparison

    def calcularValores(self, ds,n_clas):
        # # algorithms, # statistics
        algorithms = self.setAlgorithms()
        mat = np.zeros((len(algorithms),8))

        for alg in range(len(algorithms)):              
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
        algorithms = self.setAlgorithms()
        wil = np.zeros(len(algorithms))
        alg_best= mat.argmax()

        better =self.__load(ds,n_clas, alg_best)
        
        

        better = better[:,index_metric]
        
        better_avg = np.average(better)
        wil[alg_best] = 2
        for alg in range(len(algorithms)):
            if alg_best !=alg:

                x =self.__load(ds,n_clas, alg)       

                if better_avg==np.average(x[:,index_metric]):
                    wil[alg] = 2
                else:
                    statistic, pvalue = wilcoxon(better,x[:,index_metric])
                    if pvalue>=0.05:
                        wil[alg] = 1

        return wil
