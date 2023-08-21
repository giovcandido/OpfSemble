from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from opfython.models.supervised import SupervisedOPF
from ensemble_item import EnsembleItem
from copy import deepcopy
import numpy as np
import sys
import os
import pickle

import logging
logging.disable(sys.maxsize)

class Ensemble:
    """
    A class which creates the models for the ensemble learning.
    """

    def __init__(self, n_models=10, loading_path=None, saving_path=None):
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

        self.n_models=n_models
        self.items = []
        
        if (loading_path != None):            
            self.load_models(loading_path)
            return
        
        if (not type(n_models) == int):
            raise Exception('The number of models must be an integer value!')
    
        base_models_names = ['KNN', 'SVM', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'MLP']
#        self.n_models=n_models
#        self.items = []

        #print('Creating model  1')
        self.items.append(EnsembleItem('OPF_1',SupervisedOPF()))
        #print('Creating model  2')
        self.items.append(EnsembleItem('Naive Bayes_1',GaussianNB()))
    
        for i in range(n_models-2):
            #print('Creating model ', i+3)
            model = base_models_names[np.random.randint(0, len(base_models_names))]
            keys=[]
            for item in self.items:
                keys.append(item.key)

            id_model = len([key for key in keys if key.startswith(model)])
            
            if (model == 'KNN'):
                self.items.append(EnsembleItem(model + '_' + str(id_model + 1),KNeighborsClassifier(n_neighbors = np.random.randint(1, 51))))
            elif(model == 'SVM'):
                self.items.append(EnsembleItem(model + '_' + str(id_model + 1),SVC(C=np.random.random(),
                                                             kernel=np.random.choice(['rbf','linear','poly','sigmoid'], 1)[0],
                                                             degree=np.random.randint(1, 10),
                                                             gamma=np.random.random())))
            elif(model == 'Random Forest'):
                self.items.append(EnsembleItem(model + '_' + str(id_model + 1),RandomForestClassifier(n_estimators=np.random.randint(10,500),
                                                                                   criterion=np.random.choice(['gini', 'entropy'], 1)[0])))


            elif(model == 'Gradient Boosting'):
                self.items.append(EnsembleItem(model + '_' + str(id_model + 1),GradientBoostingClassifier(learning_rate=np.random.random_sample(),
                                                                                       n_estimators=np.random.randint(10,500))))

            elif(model == 'Extra Trees'):
                self.items.append(EnsembleItem(model + '_' + str(id_model + 1),ExtraTreesClassifier(n_estimators=np.random.randint(10,500),
                                                                                 criterion=np.random.choice(['gini', 'entropy'], 1)[0])))
            elif(model == 'MLP'):
                self.items.append(EnsembleItem(model + '_' + str(id_model + 1),MLPClassifier(hidden_layer_sizes=(np.random.randint(10,300),),
                															activation=np.random.choice(['logistic', 'tanh', 'relu'], 1)[0],
                															solver=np.random.choice(['lbfgs', 'sgd', 'adam'], 1)[0],max_iter=300,
                															early_stopping=True )))                                                                
                                                                                 
            
        if (saving_path != None):
            self.save_models(os.path.join(saving_path, str(n_models)))
            
    def save_models(self, path):
        if (len(self.items) == 0):
            raise Exception('No list of models has been created yet!')
        
        # Create dir if it does not exists
        if (not os.path.exists(path)):
            os.makedirs(path)
        
        for item in self.items:
            file_name = os.path.join(path, item.key+'.sav') # Get the name of the model
            pickle.dump(item.classifier, open(file_name, 'wb')) # save the model
            
    def load_models(self, path):
        if (not os.path.exists(path)):
            raise Exception('The given path is not invalid!')
        
        models = os.listdir(path)
        
        for model in models:
            m = deepcopy(pickle.load(open(os.path.join(path,model), 'rb')))

            # limit the number of iterations of the SVM model with polynomial kernel
            # it is important to avoid an 'endless' fitting of the model
            if (isinstance(m,SVC) and m.kernel=='poly'):
                m.max_iter=100000

            self.items.append(EnsembleItem(model.split('.')[0], m))
