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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from ensemble import Ensemble
from ensemble_item import EnsembleItem
import numpy as np

class SuperLearner:
	"""
	A class whose the functions of the training and testing of the super learner algorithm
	are implemented
	"""

	def __init__(self, models=None, n_models=10, n_folds=10, type_='all'):
		"""
		Initialization of the following properties:
			- models: a list with the base-models
			- meta_model: points to the metal-model
	
		Parameters
		----------
		models: list
			A list that contains the base learning models. Default is None, which means
			that the following default base learning models will be created: Logistic Regression,
			KNN, Gradient Boosting, Random Forest and SVM
		"""        
	
		if (type_ !='all' and type_ != 'first'):
			raise Exception('Type must be \'all\' or \'first\'')
			
		ens = None            
	
		if (models == None):
			ens = Ensemble(n_models=n_models)
		else:
			ens = models
			
		if (type_ == 'all'):
			self.ensemble = ens
		else:            
			ens_ = []
			
			for item in ens.items:
				if (item.key.endswith('_1')):
					ens_.append(EnsembleItem(item.key,item.classifier))


			ens.items = ens_
			ens.n_models = len(ens_)
		for item in ens.items:
			item.score=0.0
			item.cluster_id=0

		self.ensemble = ens
		self.n_folds = n_folds        
		self.meta_model = None        
		
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
		
		meta_X, meta_y = list(), list()
		# define split of data
		kfold = KFold(n_splits=self.n_folds, shuffle=True)
		
		# enumerate splits
		for train_ix, test_ix in kfold.split(X):
		    fold_yhats = list()
		    # get data
		    train_X, test_X = X[train_ix], X[test_ix]
		    train_y, test_y = y[train_ix], y[test_ix]
		    meta_y.extend(test_y)
		    # fit and make predictions with each sub-model
		    
		    for item in self.ensemble.items:
		        model = item.classifier
		        model.fit(train_X, train_y)
		        yhat = np.asarray(model.predict(test_X)).reshape(-1,1)
		        # store columns
		        fold_yhats.append(yhat)
		                
		    # store fold yhats as columns
		    meta_X.append(hstack(fold_yhats))
		
		meta_X = vstack(meta_X)
		meta_y = asarray(meta_y)
		
		self.fit_meta_model(meta_X, meta_y)
		#return vstack(meta_X), asarray(meta_y)

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
		
		for item in self.ensemble.items:
		    model = item.classifier
		    yhat = np.asarray(model.predict(X)).reshape(-1,1)
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
		
		self.meta_model = LogisticRegression()
		self.meta_model.fit(X, y)
