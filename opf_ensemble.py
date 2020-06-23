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
import numpy as np
import sys

import logging
logging.disable(sys.maxsize)

class OpfSemble:
    """
    A class which implements the OPF Ensemble Learning.
    """

    def __init__(self, n_models=10, n_folds=10):
        """
        Initialization of the class properties.
    
        Parameters
        ----------
        n_models: int
            - The number of models to be created. Default is 10
        n_folds: int
            - The number of folds of the cross validation
        """

        self.ensemble = Ensemble(n_models=n_models).ensemble
        self.n_folds = n_folds
        self.n_models = n_models
        self.prototypes = None
    
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
    
        # define split of data
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
    
        # enumerate splits
        for train_ix, test_ix in kfold.split(X):
            fold_res = list()
            # get data
            train_X, test_X = X[train_ix], X[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            # fit and make predictions with each sub-model
            
            for key in self.ensemble:
                model = self.ensemble[key]
                model.fit(train_X, train_y)
                yhat = model.predict(test_X)
                res = (yhat == test_y).astype(int)
                # store columns
                fold_res.append(res)
            
            # store fold yhats as columns
            meta_X = np.hstack([meta_X, vstack(fold_res)]) if meta_X.size else vstack(fold_res)
    
        return meta_X


    def predict(self, X):
        """
        Perform the classification of the data in X with stacked model
    
        Parameters
        ----------
        X: array
            A 2D array with the testing/validation set
    
        Returns
        -------
        predictions: array
            A column array with the classification results of each sample in X
        """

        if self.prototypes is None:
            raise Exception('Meta model was not fitted!')
        preds = []
        for key in self.prototypes:
            model = self.prototypes[key]
            preds.append(model.predict(X))
        preds = np.asarray(preds)
        pred, _ = mode(preds, axis=0)

        return pred[0]

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
        for i in range(opf_unsup.subgraph.n_nodes):
            if opf_unsup.subgraph.nodes[i].idx==opf_unsup.subgraph.nodes[i].root:        
                proto.append(opf_unsup.subgraph.nodes[i].root)

        prototype = dict()
        i = 0
        for key in self.ensemble:
            for j in range(len(proto)):
                if(i==proto[j]):        
                    prototype[key] = self.ensemble[key]
            i+=1
        self.prototypes =  prototype
