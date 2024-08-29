import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from opf_ensemble import OpfSemble

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)

def load_data(data_path='./dataset.csv'):
    # Load the dataset
    dataset = pd.read_csv(data_path)

    # Get the features
    features = dataset.loc[:, 'Espécie - Caesalpinia ferrea var. leiostachya - One-Hot':'Copa Desequilibrada'].to_numpy()

    # Get the targets
    targets = dataset['Nível de Deterioração - Colo - 3 Classes'].to_numpy()

    # Return the loaded data
    return features, targets

execs = 1
mean = 0.

for seed in range(execs):
    np.random.seed(seed)

    # Load the dataset
    # X, y = load_breast_cancer(return_X_y=True)
    X, y = load_data()

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size=.3)

    # List of pre-defined classifiers
    classifiers = [ExtraTreesClassifier(n_estimators=50)] * 30

    # Build the OPFsemble instance
    # ens = OpfSemble(ensemble=classifiers, meta_data_mode='oracle', divergence=None, bootstrapping=False)
    ens = OpfSemble(ensemble=classifiers, meta_data_mode='count_class', divergence=None, bootstrapping=False)

    # Get the meta data
    meta_X = ens.fit(X_train, y_train)

    # Fit the model with the meta data in order to find the clusters of similar classifiers
    ens.fit_meta_model(meta_X, k_max=5)

    # Get the predictions
    # y_pred = ens.predict(X_test, voting='hard_mode')
    y_pred = ens.predict(X_test, voting='soft_mode')

    # print(ens.clusters)
    # print(ens.prototypes)

    # for item in ens.ensemble.items:
    # 	print(item.key)

    mean += accuracy_score(y_test, y_pred)

print('Accuracy: ', mean / execs)
