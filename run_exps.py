from opf_ensemble import OpfSemble
from super_learner import SuperLearner
from ensemble import Ensemble
from time import time
from sklearn.metrics import accuracy_score,f1_score
from sklearn.ensemble import StackingClassifier
from copy import deepcopy
import os
import sys
import math
import sklearn
import numpy as np
import warnings
import argparse

# Disable all types of warning
warnings.filterwarnings("ignore")

def get_main_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-i', type=str, help='Folder where the dataset is located.')
    parser.add_argument('-m', type=str, help='Folder where the base models are located.')
    parser.add_argument('-o', type=str, help='Folder where the results will be save.')

    return parser.parse_args()

def dict2array(dictionary):
    '''
        Function that transforms a dictionary into a numpy array in a following format:
        
        #################################
        #Model     Accuracy     F1-score
        Model_1    Acc_value1   F1_value1
        Model_2    Acc_value2   F1_value2
        .
        .
        .
        Model_n    Acc_valuen   F1_valuen
        #################################

        Parameters
        ----------
            dictionary: dict
                A dictionary in which the keys are the models' name, and the values are a list with two positions: Accuracy and F1 values
    '''

    final_list = []
    for key in dictionary:
        aux = []
        model = key
        aux.append(model)
        arr = np.array(dictionary[key])

        for j in range(len(arr)):
            aux.append(arr[j])

        final_list.append(aux)

    return np.array(final_list,dtype=object)

def build_stacking_classifier(ensemble,n_folds):
    '''
        Function that creates a Stacking Generalization model for classification.

        Parameters
        ----------
        ensemble : Ensemble
            Ensemble object. Each element is an Ensemble object composed of a key (model's name) and the model itself.
        n_folds: int
            Number of cross validation folds to construct the meta-data of the stacking generalization.

        Returns
        -------
        stacking : SuperLearner
            A Stacking Classifier object
    '''

    estimators = []

    for e in ensemble:
        estimators.append((e.key,deepcopy(e.classifier)))
    
    return SuperLearner(models=estimators,n_folds=n_folds)

if __name__ == '__main__':
    args = get_main_args()
    data = args.i
    models_path = args.m
    results_folder = args.o

    # Check if the data folder exists
    if (not os.path.exists(data)):
        sys.exit('Data folder does not exists. Please check if the path to the data is correct or already exists.')

    # Reading the datasets' folders
    #ds = [d for d in os.listdir(data) if os.path.isdir('{}/{}'.format(data,d))]
    ds = ['iris']

    # Auxiliary variables
    folds = output_folder = y_pred = meta_X = None
    start_time_opf = end_time_opf = 0
    opf_variants = ['mode','intercluster','intracluster','mode_best','aggregation'] # OPFsemble variants
    divergence = {'nodivergence':None,'yule':'yule','disagreement':'disagreement'} # The divergence metrics
    meta_data_type = {'oracle':'oracle','countclass':'count_class'} # The meta-data type of the classifiers predictions
    n_folds_ensemble = 10 # Number of folds to construct the meta-data from the baseline classifiers
    k_max_list = [2,5,10,20,30] # k_max values for optimization
    opf_ens = OpfSemble() # OPFsemble instance

    # Performs the experiments for each meta data type
    for meta in meta_data_type:    
        # Performs the experiments for each divergence mode
        for dv in divergence:
            # Performs the experiments for each dataset
            for d in ds:
                # For each cross validation fold
                folds = os.listdir('{}/{}'.format(data,d))
                print('Number of folds for the dataset {}: {}'.format(d,len(folds)))

                for f in folds:
                    # Loading the training, validation and test sets
                    train = np.loadtxt('{}/{}/{}/train.txt'.format(data,d,f),delimiter=',')
                    valid = np.loadtxt('{}/{}/{}/valid.txt'.format(data,d,f),delimiter=',')
                    test = np.loadtxt('{}/{}/{}/test.txt'.format(data,d,f),delimiter=',')

                    # Split the data into X and y
                    # The last column is the output variable to be predicted
                    X_train,X_valid,X_test = train[:,:-1],valid[:,:-1],test[:,:-1]
                    y_train,y_valid,y_test = train[:,-1].astype(int),valid[:,-1].astype(int),test[:,-1].astype(int)

                    # For each number of baseline classifiers
                    for n in [10,30,50]:
                        print('META TYPE {}, DATASET {}, FOLD {}, NUMBER OF CLASSIFIERS {} '.format(meta,d,f,n))
                        # Loading the baseline models
                        if (models_path != None):
                            print('Loading the baseline classifier models...')
                            ens = Ensemble(n_models=n,loading_path='{}/{}'.format(models_path,n))
                        else:
                            ens = None

                        # Assigns the OPFsemble's parameters
                        opf_ens.n_models=n
                        opf_ens.n_folds=n_folds_ensemble
                        opf_ens.divergence=divergence[dv]
                        opf_ens.ensemble = ens
                        opf_ens.meta_data_mode = meta

                        # Creates the stacking generalization model
                        stacking = SuperLearner(deepcopy(opf_ens.ensemble),n_folds=n_folds_ensemble)

                        # Training the OPFsemble
                        print('Building the meta-data of the OPFsemble....')
                        start_time_opf = time()
                        meta_X = opf_ens.fit(X_train,y_train)
                        end_time_opf = time() -start_time_opf                

                        # Gets the baseline model's predictions
                        # Check if the baseline model's folder results already exists
                        output_folder = '{}/{}/{}/{}/{}'.format(results_folder,'baseline',d,f,n)
                        if (not os.path.exists(output_folder)):
                            os.makedirs(output_folder)
                        
                        if (not os.path.exists('{}/results.txt'.format(output_folder))):
                            print('Performing the test with the baseline models...')
                            # Getting the baseline predictions as array
                            scores_baselines = dict2array(opf_ens.get_scores_baselines(X_test,y_test))
                            # Saving the baseline's results y_pred
                            np.savetxt('{}/results.txt'.format(output_folder),scores_baselines,fmt='%s',delimiter=',',header='Model,Accuracy,F1')
                        else:
                            print('Folder {} already exists with all the validation metrics...'.format(output_folder))

                        
                        # Test with the stacking generalization model
                        # Check if the stacking generalization folder results already exists
                        output_folder = '{}/{}/{}/{}/{}'.format(results_folder,'stacking_generalization',d,f,n)
                        if (not os.path.exists(output_folder)):
                            os.makedirs(output_folder)

                        if (not os.path.exists('{}/results.txt'.format(output_folder))):
                            print('Performing the test with the stacking generalization...')
                            # Getting the stacking generalization results
                            start_time_stack = time()
                            stacking.fit(X_train,y_train)
                            end_time_stack = time() - start_time_stack
                            y_pred = stacking.predict(X_test)
                            # Computing accuracy and f1-score
                            acc = accuracy_score(y_test,y_pred)
                            f1 = f1_score(y_test,y_pred,average='weighted')
                            # Saving the validation measures and y_pred
                            np.savetxt('{}/results.txt'.format(output_folder),np.array([acc,f1,end_time_stack]),fmt='%.4f',delimiter=',',header='Accuracy,F1-score,Fit time')
                            np.savetxt('{}/y_pred.txt'.format(output_folder),y_pred,fmt='%.4f',delimiter=',')
                        else:
                            print('Folder {} already exists with all the validation metrics...'.format(output_folder))

                        
                        # Test with the OPFsemble and its variants
                        for v in opf_variants:
                            print('OPF variant: ',v)
                            # Check if the output folder exists
                            if (meta=='count_class'):
                                output_folder = '{}/{}/{}/{}/{}/{}'.format(results_folder,'opf_{}'.format(meta),v,d,f,n)
                            else:
                                output_folder = '{}/{}/{}/{}/{}/{}'.format(results_folder,'opf_{}'.format(dv),v,d,f,n)
                            if (not os.path.exists(output_folder)):
                                os.makedirs(output_folder)
                            
                            if(os.path.exists('{}/results.txt'.format(output_folder))):
                                print('Folder {} already exists. Moving to the next OPF variant...'.format(output_folder))
                                continue

                            # Seeks the best value for k_max using the validation set
                            best_k_max = None
                            highest_f1 = -1
                            k_max_valid = []
                            for k_max in k_max_list:
                                start_time_unsup = time()
                                opf_ens.fit_meta_model(meta_X,k_max)
                                end_time_unsup = time() - start_time_unsup

                                y_pred = opf_ens.predict(X_valid,voting=v)
                                f1 = f1_score(y_valid,y_pred,average='weighted')

                                if (f1 > highest_f1):
                                    highest_f1 = f1
                                    best_k_max = k_max

                                k_max_valid.append([k_max,f1,end_time_unsup])

                            # Saving the tested k_max values and their F1 scores
                            np.savetxt('{}/k_max_validation.txt'.format(output_folder),np.array(k_max_valid),fmt='%.4f',delimiter=',',header='k_max,F1,Meta model time')

                            # OPFsemble predictions using the test set and the best k_max value
                            opf_ens.fit_meta_model(meta_X,k_max=best_k_max)
                            y_pred = opf_ens.predict(X_test,voting=v)
                            # Computing accuracy and f1-score
                            acc = accuracy_score(y_test,y_pred)
                            f1 = f1_score(y_test,y_pred,average='weighted')
                            # Saving the validation measures, the meta_X and y_pred
                            np.savetxt('{}/y_pred.txt'.format(output_folder),y_pred,fmt='%.4f',delimiter=',')
                            np.savetxt('{}/meta_X.txt'.format(output_folder),meta_X,fmt='%.4f',delimiter=',')
                            np.savetxt('{}/results.txt'.format(output_folder),np.array([acc,f1,end_time_opf]),fmt='%.4f',delimiter=',',header='Accuracy,F1,Fit time')                    
                            # Saving the clusters and their prototypes
                            opf_ens.save_clusters(output_folder)