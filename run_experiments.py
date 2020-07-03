from opf_ensemble import OpfSemble
from super_learner import SuperLearner
from time import time
import numpy as np
import sys
import os
import copy

from sklearn.metrics import accuracy_score, f1_score
np.set_printoptions(threshold=sys.maxsize)

if len(sys.argv) <= 1:
    print('Usage: %s <dataset name>' % sys.argv[0])
    raise SystemExit

def validation(classifiers_feats, X_valid, y_valid, voting='mode'):
    k_max = [5,10,20,30,40,50]
    best_k = 0
    value_best_k = -1.0

    results_validation=[]
    for k in k_max:
        opf_ens.fit_meta_model(classifiers_feats,k)
        accuracy, f1 = computeMetrics(opf_ens.predict(X_valid, voting= voting),y_valid)
        results_validation.append([k, accuracy, f1])
        if f1>value_best_k:
            best_k = k
            value_best_k = f1        
    return best_k, results_validation

def run(classifiers_feats, X_test, X_valid, y_valid, voting='mode'):
    best_k, results_validation = validation(classifiers_feats, X_valid, y_valid, voting)
    opf_ens.fit_meta_model(classifiers_feats,best_k)
    return opf_ens.predict(X_test, voting), best_k, results_validation

def saveResults(pred_ensamble, X_train, y_train, y_test, best_k, validation, exec_time, path, obj, ensemble_name='OPF_ENSEMBLE', compute_models=False):
    results = ''
    accuracy, f1 = computeMetrics(pred_ensamble,y_test)
    results = results + '{:s},{:d},{:.4f},{:.4f},{:.4f}\n'.format(ensemble_name, best_k, accuracy, f1, exec_time)

    if compute_models:
        for item in obj.items:
            model = item.classifier
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            accuracy, f1 = computeMetrics(pred,y_test)
            results = results + '{:s},{:d},{:.4f},{:.4f},{:.4f}\n'.format(item.key,0, accuracy, f1, 0)

    np.savetxt('{}/pred.txt'.format(path), pred, fmt='%d')
    np.savetxt('{}/validation.txt'.format(path), validation, fmt='%d,%.5f,%.5f')
    output= open('{}/results.txt'.format(path), "w")    
    output.write(results)
    print('Results:')
    print('    {}/pred.txt'.format(path))
    print('    {}/results.txt'.format(path))
    print('    {}/validation.txt'.format(path))

def computeMetrics(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, f1


datasets = ['vertebral_column']
n_models = [10,30]#,50,100]

#ds = sys.argv[1]

results_folder = 'Results_test'


for ds in datasets:
	for n in n_models:
		for f in range(1,2):
		    
		    train = np.loadtxt('data/{}/{}/train.txt'.format(ds,f),delimiter=',', dtype=np.float32)
		    valid = np.loadtxt('data/{}/{}/valid.txt'.format(ds,f),delimiter=',', dtype=np.float32)
		    test = np.loadtxt('data/{}/{}/test.txt'.format(ds,f),delimiter=',', dtype=np.float32)

		    X = train[:,:-1]
		    y = train[:,-1].astype(np.int) 

		    X_valid = valid[:,:-1]
		    y_valid = valid[:,-1].astype(np.int) 

		    X_test = test[:,:-1]
		    y_test = test[:,-1].astype(np.int) 

		    concat = np.concatenate((y, y_valid))
		    concat = np.concatenate((concat, y_test))
		    n_classes = len(np.unique(concat))
		    if n_classes<np.max(concat):
		        n_classes = np.max(concat)

		    start_time = time()
		    opf_ens = OpfSemble(n_models=n, n_folds=10, n_classes=n_classes)
		    new_x = opf_ens.fit(X, y)
		    end_time_initial = time() -start_time
		    voting = ['mode', 'average', 'intercluster','mode_best']
		    for vote in voting:

		        ResultsPath = '{}/OPF_Ensemble_{}/{}/{}/{}'.format(results_folder,vote,ds,f,n)
		        if not os.path.exists(ResultsPath):
		            os.makedirs(ResultsPath)

		        start_time = time()        
		        pred_ensamble, best_k, validation_results = run(new_x, X_test, X_valid, y_valid, voting = vote)
		        end_time = time() -start_time
		        saveResults(pred_ensamble, X, y, y_test, best_k, validation_results, end_time+end_time_initial, ResultsPath, opf_ens.ensemble, ensemble_name='OPF_ENSEMBLE_{}'.format(vote), compute_models=True)  

		    ResultsPath_Sl = '{}/Super_Learner/{}/{}/{}'.format(results_folder,ds,f,n)
		    if not os.path.exists(ResultsPath_Sl):
		        os.makedirs(ResultsPath_Sl)

		    start_time = time()
		    sl = SuperLearner(models=copy.deepcopy(opf_ens.ensemble), n_folds=10, type_='first')
		    sl.fit(X, y)
		    preds_super_learner = sl.predict(X_test)
		    end_time = time() - start_time

		    saveResults(preds_super_learner, X, y, y_test, 0, np.asarray([[0, 0.0, 0.0]]), end_time, ResultsPath_Sl, sl.ensemble, ensemble_name='Super_Learner', compute_models=True)

		    ResultsPath_Sl_Full = '{}/Super_Learner_Full/{}/{}/{}'.format(results_folder,ds,f,n)
		    if not os.path.exists(ResultsPath_Sl_Full):
		        os.makedirs(ResultsPath_Sl_Full)

		    start_time = time()
		    sl_full = SuperLearner(models=copy.deepcopy(opf_ens.ensemble), n_folds=10)
		    sl_full.fit(X, y)
		    preds_super_learner_full = sl_full.predict(X_test)
		    end_time = time() - start_time

		    saveResults(preds_super_learner_full, X, y, y_test, 0, np.asarray([[0, 0.0, 0.0]]), end_time, ResultsPath_Sl_Full, sl_full.ensemble, ensemble_name='Super_Learner_Full', compute_models=True)
