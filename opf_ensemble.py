# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:04:58 2020

@author: DANILO

THIS CODE WAS ADAPTED FROM THE EXAMPLE AVAILABLE AT https://machinelearningmastery.com/super-learner-ensemble-in-python/
AND CREATED BY Jason Brownlee
"""

from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from opfython.models.supervised import SupervisedOPF
import numpy as np

class OpfSemble:
    """
    A class whose the functions of the training and testing of the super learner algorithm
    are implemented
    """
    
    def __init__(self, n_models=10, n_folds=10):
        """
        Initialization of the following properties:
            - n_models: the number of models to be created
            - n_folds: number of folds of the cross validation
        
        Parameters
        ----------
        models: list
            A list that contains the base learning models. Default is None, which means
            that the following default base learning models will be created: Logistic Regression,
            KNN, Gradient Boosting, Random Forest and SVM
        """

        if (not type(n_models) == int):
            raise Exception('The number of models must be an integer value!')
        
        base_models_names = ['KNN', 'SVM', 'OPF', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'LDA']
        ensemble = dict()
        
        for i in range(n_models):
            print('Creating model ', i+1)
            model = base_models_names[np.random.randint(0, len(base_models_names))]
            id_model = len([key for key in ensemble.keys() if key.startswith(model)])
            
            if (model == 'KNN'):
                ensemble[model + '_' + str(id_model + 1)] = KNeighborsClassifier(n_neighbors = np.random.randint(1, 51))
            elif(model == 'SVM'):
                ensemble[model + '_' + str(id_model + 1)] = SVC(C=np.random.random(),
                                                             kernel=np.random.choice(['rbf','linear','poly','sigmoid'], 1)[0],
                                                             degree=np.random.randint(1, 10),
                                                             gamma=np.random.random())
            elif(model == 'OPF'):
                ensemble[model + '_' + str(id_model + 1)] = SupervisedOPF()
            elif(model == 'Random Forest'):
                ensemble[model + '_' + str(id_model + 1)] = RandomForestClassifier(n_estimators=np.random.randint(10,500),
                                                                                   criterion=np.random.choice(['gini', 'entropy'], 1)[0])
            elif(model == 'Gradient Boosting'):
                ensemble[model + '_' + str(id_model + 1)] = GradientBoostingClassifier(learning_rate=np.random.random_sample(),
                                                                                       n_estimators=np.random.randint(10,500))
            elif(model == 'Extra Trees'):
                ensemble[model + '_' + str(id_model + 1)] = ExtraTreesClassifier(n_estimators=np.random.randint(10,500),
                                                                                 criterion=np.random.choice(['gini', 'entropy'], 1)[0])
            elif(model == 'LDA'):
                ensemble[model + '_' + str(id_model + 1)] = LinearDiscriminantAnalysis(solver=np.random.choice(['svd','lsqr','eigen'], 1)[0])
        
        self.ensemble = ensemble
        self.n_folds = n_folds
        self.n_models = n_models
        
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
        
        meta_X = list()
        
        for model in self.models:
            yhat = model.predict_proba(X)
            meta_X.append(yhat)
    	
        meta_X = hstack(meta_X)
    	
        # predict
        return self.meta_model.predict(meta_X)
    
    # fit a meta model
    def fit_meta_model(self, X, y):
        """
        Perform the training of the meta-model
        
        Parameters
        ----------
        X: array
            A 2D array with the training set
        y: array
            A column array with the labels of each samples in X 
        """
        
        self.meta_model = RandomForestClassifier(n_estimators=10, criterion='entropy')
        self.meta_model.fit(X, y)   