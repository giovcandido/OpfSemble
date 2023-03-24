# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 22:38:56 2020

@author: DANILO
"""

from ensemble import Ensemble

n_models = [10,30,50,100]

for n in n_models:
    ensemble = Ensemble(n_models=n,saving_path='models')