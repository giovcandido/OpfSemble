from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from opfython.models.supervised import SupervisedOPF
import numpy as np
import sys

import logging
logging.disable(sys.maxsize)

class Ensemble:
	"""
	A class which creates the models for the ensemble learning.
	"""

	def __init__(self, n_models=10):
		"""
		Initialization of the following properties:
			- n_models: the number of models to be created
	
		Parameters
		----------
		models: list
			A list that contains the base learning models. Default is None, which means
			that the following default base learning models will be created: Logistic Regression,
			KNN, Gradient Boosting, Random Forest and SVM
		"""

		if (not type(n_models) == int):
			raise Exception('The number of models must be an integer value!')
	
		base_models_names = ['KNN', 'SVM', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'LDA']
		ensemble = dict()

		#print('Creating model  1')
		ensemble['OPF_1'] = SupervisedOPF()
		#print('Creating model  2')
		ensemble['Naive Bayes_1'] = GaussianNB()
	
		for i in range(n_models-2):
			#print('Creating model ', i+3)
			model = base_models_names[np.random.randint(0, len(base_models_names))]
			id_model = len([key for key in ensemble.keys() if key.startswith(model)])
			
			if (model == 'KNN'):
			    ensemble[model + '_' + str(id_model + 1)] = KNeighborsClassifier(n_neighbors = np.random.randint(1, 51))
			elif(model == 'SVM'):
			    ensemble[model + '_' + str(id_model + 1)] = SVC(C=np.random.random(),
			                                                 kernel=np.random.choice(['rbf','linear','poly','sigmoid'], 1)[0],
			                                                 degree=np.random.randint(1, 10),
			                                                 gamma=np.random.random())
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