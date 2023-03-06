# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:04:58 2020

@author: DANILO

THIS CODE WAS ADAPTED FROM THE EXAMPLE AVAILABLE AT https://machinelearningmastery.com/super-learner-ensemble-in-python/
AND CREATED BY Jason Brownlee
"""

from numpy import vstack
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.stats import mode
from opfython.models.unsupervised import UnsupervisedOPF
from ensemble import Ensemble
from sklearn.metrics import f1_score
from collections import defaultdict
import math
import numpy as np
import sys

import logging
logging.disable(sys.maxsize)

class OpfSemble:
    """
    A class which implements the OPF Ensemble Learning.
    """

    def __init__(self, n_models=10, n_folds=10,n_classes=0,ensemble=None,divergence=False):
        """
        Initialization of the class properties.

        Parameters
        ----------
        n_models: int
            - The number of models to be created. Default is 10
        n_folds: int
            - The number of folds of the cross validation
        """
        
        if (ensemble != None):
            self.ensemble = ensemble
        else:        
            self.ensemble = Ensemble(n_models=n_models)
        
        self.n_models = n_models
        self.n_folds = n_folds
        self.prototypes = None
        self.clusters = None
        self.prototypes_scores = None
        self.n_classes = n_classes

        self.ensemble
        self.divergence = divergence

    # create a list of base-models
    def get_models(self):
        """
        Creates and returns a list with the default base-models.

        Returns
        -------
        models: list
            A list with the base-models
        """

        models = list()
        models.append(LogisticRegression(max_iter=22))
        models.append(KNeighborsClassifier(n_neighbors=1))
        models.append(GradientBoostingClassifier(learning_rate=0.09, n_estimators=500))
        models.append(RandomForestClassifier(n_estimators=10, criterion='entropy'))
        models.append(SVC(C=61, gamma=0.08, kernel='rbf', degree=1, probability=True))

        return models

    def fit(self, X, y):
        """
        Training of the base-models and the meta-model

        Parameters
        ----------
        X: array
            A 2D array with the training set
        y: array
            A column array with the labels of each samples in X        
        """

        meta_X = np.array([], dtype=int)
        if self.n_classes==0:
            self.n_classes = len(np.unique(y))
            if self.n_classes<np.max(y):
                self.n_classes = np.max(y)

        # define split of data
        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        # enumerate splits
        for train_ix, test_ix in kfold.split(X):
            fold_res = list()
            # get data
            train_X, test_X = X[train_ix], X[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            # fit and make predictions with each sub-model
        
            for item in self.ensemble.items:
                model = item.classifier
                model.fit(train_X, train_y)
                yhat = model.predict(test_X)
                item.score += f1_score(yhat, test_y, average='weighted')/self.n_folds
                #res = (yhat == test_y).astype(int)
                # store columns
                fold_res.append(yhat)
        
            # store fold yhats as columns
            meta_X = np.hstack([meta_X, vstack(fold_res)]) if meta_X.size else vstack(fold_res)

        # Creating a new vector with the baseline predictions' counts for each class
        new_x = np.zeros((self.n_models,self.n_classes))
        for i,_ in enumerate(meta_X):
            un,counts = np.unique(meta_X[i],return_counts=True)
            for j,c in enumerate(un):
                # Minimum label value must be 0 for the new_x indexing
                c = c if np.min(un) == 0 else c - 1
                new_x[i,c] = counts[j]
        
        # Check if Kullback-Lieber divergence should be calculated for the meta_X
        if self.divergence:
            new_x = self.__calculate_divergence(new_x)
            
        return new_x

    def predict(self, X, voting='mode'):
        """
        Perform the classification of the data in X with stacked model

        Parameters
        ----------
        X: array
            A 2D array with the testing/validation set
        voting: array
            A string representing the voting approach (either mode or average)

        Returns
        -------
        predictions: array
            A column array with the classification results of each sample in X
        """

        if self.prototypes is None:
            raise Exception('Meta model was not fitted!')
            
        if self.clusters is None:
            raise Exception('No cluster is defined! It might be happened because the model is not fitted yet!')

        if voting=='intracluster':
            # Predictions from the classifiers belonging to the protopype with the highest F1-score
            
            # Getting the key of the best-performing prototype, i.e., the prototype with the highest F1-score
            max_key = max(self.prototypes_scores, key=self.prototypes_scores.get)
            
            # Getting the ID of the best-performing prototype assigned by the Unsupervised OPF
            idx_prot = np.where([i.key == max_key for i in self.ensemble.items])[0][0]
            
            # Getting the classifiers from the best-performing cluster
            clfs = self.clusters[idx_prot]
            
            # Getting predictions from classifiers
            preds = []
            for c in clfs:
                preds.append(c.predict(X))
            
            # Getting the final predictions based on the most common predicted class among the classifiers
            pred = mode(np.asarray(preds),axis=0)[0].reshape(-1,1)  
        else:
            preds = []
            scores = []
    
            for key in self.prototypes:
                model = self.prototypes[key]
                scores.append(self.prototypes_scores[key])
                pr = model.predict(X)
                preds.append(pr)
    
            scores = np.asarray(scores)
            preds = np.asarray(preds)
    
            if voting=='mode':
                #major voting
                pred, _ = mode(preds, axis=0)
                pred = pred[0]
            elif voting=='average':
                #compute the prototypes average score by label
                mat = np.zeros((self.n_classes, len(self.prototypes)))
                pred = np.zeros(len(X))
                for sample in range(len(X)):
                    for prot in range(len(self.prototypes)):
                        mat[preds[prot,sample]-1,prot] = scores[prot]
                    
                    summing = np.zeros(len(self.prototypes))
                    count = np.zeros(len(self.prototypes))
                    
                    for prot in range(len(self.prototypes)):
                        if mat[preds[prot,sample]-1,prot]>0:
                            summing[prot]+=scores[prot]
                            count[prot]+=1
                    pred_avgs = np.zeros(len(self.prototypes))
                    
                    for prot in range(len(self.prototypes)):
                        pred_avgs[prot] = summing[prot]/count[prot]
    
                    pred[sample] = preds[np.argmax(pred_avgs),sample]
            elif voting=='intercluster':
                preds_each_cluster = list()
                
                for key in self.clusters: # iterate in each cluster to obtain its classifiers
                    preds = list()
                    
                    for model in self.clusters[key]: # iterate in each model of the current cluster                  
                        preds.append(model.predict(X))
                    
                    preds = np.asarray(preds)
                    
                    # Mode for the current cluster (intracluster voting)
                    preds_each_cluster.append(mode(preds, axis=0)[0])
                            
                preds_each_cluster = np.vstack(preds_each_cluster)
                
                pred = preds_each_cluster        
                
                # Mode for all clusters (intercluster voting)
                # Minimum of two clusters to calculate the mode
                if (len(preds_each_cluster) > 1):
                    pred = mode(preds_each_cluster, axis=0)[0]
                else:
                    pred = preds_each_cluster
                
                pred = pred[0]
            elif voting=='mode_best':
                max_cluster_value = np.zeros(len(self.clusters))
                max_cluster_index = np.zeros(len(self.clusters)).astype(int)
                for i in range(len(self.ensemble.items)):
                    item = self.ensemble.items[i]
                    for j in range(len(self.clusters)):
                        if(item.cluster_id==j): 
                            if max_cluster_value[j]<item.score:
                                max_cluster_value[j]=item.score
                                max_cluster_index[j]=i
                preds_best_cluster = []
                for j in range(len(self.clusters)):
                    model = self.ensemble.items[max_cluster_index[j]].classifier
                    pr = model.predict(X)
                    preds_best_cluster.append(pr)
                preds_best_cluster= np.asarray(preds_best_cluster)
                pred, _ = mode(preds_best_cluster, axis=0)
                pred = pred[0]

        return pred

    # fit a meta model
    def fit_meta_model(self, X, k_max=20):
        """
        Perform the OPF clustering over the classifiers attributes and returns the prototype classifiers

        Parameters
        ----------
        X: array
            A 2D array with the classifiers attributes
        """
        opf_unsup = UnsupervisedOPF(max_k=k_max)
        opf_unsup.fit(X)

        proto= []
        clusters = defaultdict(list)
        
        for i in range(opf_unsup.subgraph.n_nodes):
            self.ensemble.items[i].cluster_id=opf_unsup.subgraph.nodes[i].root;
            if opf_unsup.subgraph.nodes[i].idx==opf_unsup.subgraph.nodes[i].root:        
                proto.append(opf_unsup.subgraph.nodes[i].root)
            
            # Adding models to the corresponding prototype root in order to form a dictionary of clusters
            clusters[opf_unsup.subgraph.nodes[i].root].append(self.ensemble.items[i].classifier)

        prototype = dict()
        prototypes_scores = dict()        
        for i in range(self.ensemble.n_models):
            for j in range(len(proto)):
                if(i==proto[j]):      
                    item=self.ensemble.items[i]
                    key = item.key
                    prototype[key] = item.classifier
                    prototypes_scores[key] = item.score           

        self.prototypes =  prototype        
        self.prototypes_scores =  prototypes_scores
        self.clusters = clusters

    def __calculate_divergence(self,X):
        '''
        Function that performs the Kullback-Lieber divergence between each pair of samples in X

        Parameters
        ----------
        X : array
            Array with N rows and M columns.

        Returns
        -------
        array
            An array with size NxN with the KL divergence between the pairs of samples in X.
        '''
        
        # calculate the kl divergence
        def kl_divergence(p, q):
            return sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))        

        kl_array = np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                c1, _ = np.histogram(X[i,:], bins=2)
                c2, _ = np.histogram(X[j,:], bins=2)
                p = c1 / X.shape[1]
                q = c2 / X.shape[1]
                kl_array[i,j] = kl_divergence(p,q)
        
        return kl_array