from opf_ensemble import OpfSemble
from super_learner import SuperLearner
from time import time
import numpy as np
import sys
import os

from sklearn.metrics import accuracy_score, recall_score, f1_score
from opfython.models.unsupervised import UnsupervisedOPF
np.set_printoptions(threshold=sys.maxsize)

if len(sys.argv) <= 1:
	print('Usage: %s <dataset name>' % sys.argv[0])
	raise SystemExit

def validation(classifiers_feats, X_valid, y_valid):
    k_max = [3,5,10,20,30,40,50]
    best_k = 0
    value_best_k = -1.0

    results_validation=[]
    for k in k_max:
        opf_ens.fit_meta_model(classifiers_feats,k)
        accuracy, f1 = computeMetrics(opf_ens.predict(X_valid),y_valid)
        results_validation.append([k, accuracy, f1])
        if f1>value_best_k:
            best_k = k
            value_best_k = f1        
    return best_k, results_validation

def run(classifiers_feats, X_test, X_valid, y_valid):
    best_k, results_validation = validation(classifiers_feats, X_valid, y_valid)
    opf_ens.fit_meta_model(classifiers_feats,best_k)
    return opf_ens.predict(X_test), best_k, results_validation

def saveResults(pred_ensamble,y_test, best_k, validation, exec_time, path, ensemble_name='OPF_ENSEMBLE'):

    results = ''
    accuracy, f1 = computeMetrics(pred_ensamble,y_test)
    results = results + '{:s},{:d},{:.4f},{:.4f},{:.4f}\n'.format(ensemble_name, best_k, accuracy, f1, exec_time)
    
    for key in opf_ens.ensemble:
        model = opf_ens.ensemble[key]
        pred = model.predict(X_test)
        accuracy, f1 = computeMetrics(pred,y_test)
        results = results + '{:s},{:d},{:.4f},{:.4f},{:.4f}\n'.format(key,0, accuracy, f1, 0)
    

    
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
    f1 = f1_score(y_true, y_pred)
    return accuracy, f1


#datasets = ['vertebral_column', 'diagnostic']
n_models = [10,30,50,100]

ds = sys.argv[1]


#for ds in datasets:
for n in n_models:
    for f in range(1,21):
        
        ResultsPath = 'Results/OPF_Ensemble/{}/{}/{}'.format(ds,f,n)
        if not os.path.exists(ResultsPath):
            os.makedirs(ResultsPath)
            
        ResultsPath_Sl = 'Results/Super_Learner/{}/{}/{}'.format(ds,f,n)
        if not os.path.exists(ResultsPath_Sl):
            os.makedirs(ResultsPath_Sl)

        train = np.loadtxt('data/{}/{}/train.txt'.format(ds,f),delimiter=',', dtype=np.float32)
        valid = np.loadtxt('data/{}/{}/valid.txt'.format(ds,f),delimiter=',', dtype=np.float32)
        test = np.loadtxt('data/{}/{}/test.txt'.format(ds,f),delimiter=',', dtype=np.float32)

        X = train[:,:-1]
        y = train[:,-1].astype(np.int) 

        X_valid = valid[:,:-1]
        y_valid = valid[:,-1].astype(np.int) 

        X_test = test[:,:-1]
        y_test = test[:,-1].astype(np.int) 

        start_time = time()
        opf_ens = OpfSemble(n_models=n, n_folds=10)
        new_x = opf_ens.fit(X, y)
        pred_ensamble, best_k, validation_results = run(new_x, X_test, X_valid, y_valid)
        end_time = time() -start_time

        saveResults(pred_ensamble, y_test, best_k, validation_results, end_time, ResultsPath)  
        
        start_time = time()
        sl = SuperLearner(models=opf_ens.ensemble.copy(), n_folds=10, type_='first')
        sl.fit(X, y)
        preds_super_learner = sl.predict(X_test)
        end_time = time() - start_time
        
        saveResults(preds_super_learner, y_test, 0, np.asarray([[0, 0.0, 0.0]]), end_time, ResultsPath_Sl, ensemble_name='Super_Learner')
