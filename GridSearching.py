#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 21:50:23 2018

@author: conorshirren
"""



from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
#from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
#from timeit import default_timer as timer
#from sklearn.metrics import confusion_matrix
#from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import numpy as np
#
#training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training', description = ' training emails')
#val_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/validation',description = 'validation emails')
#vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english",
#                              strip_accents="ascii",max_df=0.99,min_df=0.01)
#
#TrainM = vector.fit_transform(training_files.data)
#ValM = vector.transform(val_files.data)
#TrainBOG = TrainM.toarray()
#ValBOG = ValM.toarray()


def GridSarch_NB():
    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training',
                              description = ' training emails')
    val_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/validation',
                         description = 'validation emails')
    vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english",
                              strip_accents="ascii",max_df=0.99,min_df=0.01)

    TrainM = vector.fit_transform(training_files.data)
    ValM = vector.transform(val_files.data)
    #Testing for variation in alpha to reduce error
    param_grid = {'alpha': np.logspace(-1,10, 100) }
    GS = GridSearchCV(MultinomialNB(), param_grid)
    GS.fit(TrainM, training_files.target)
    GS.predict(ValM)
    best_alpha = GS.best_estimator_.alpha
    print("**Grid Search Result: Multinomisl Naive Bayes***")
    print("Best score:", (GS.best_score_*100), "%")
    print("Best estimator of alpha:",best_alpha)
    print("\n")
    
def GridSearch_SVM():
    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training', 
                              description = ' training emails')
    val_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/validation',
                         description = 'validation emails')
    vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english",
                              strip_accents="ascii",max_df=0.99,min_df=0.01)

    TrainM = vector.fit_transform(training_files.data)
    ValM = vector.transform(val_files.data)    
    #Testing for variation in C and 'max_iter' to reduce error
    param_grid = {'C': np.logspace(-1,10, 100),'max_iter': [10,100,1000]}
    GS = GridSearchCV(LinearSVC(), param_grid)
    GS.fit(TrainM, training_files.target)
    GS.predict(ValM)
    best_C = GS.best_estimator_.C
    best_max_iter = GS.best_estimator_.max_iter
    print("**Grid Search Result: Linear Support Vector Machine***")
    print("Best score:",(GS.best_score_*100), "%")
    print("Best estimator of C:",best_C )
    print("Best estimator of max_iter:", best_max_iter )
    print("\n")    
    
def GridSearch_LR():
    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training',
                              description = ' training emails')
    val_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/validation',
                         description = 'validation emails')
    vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english",
                              strip_accents="ascii",max_df=0.99,min_df=0.01)

    TrainM = vector.fit_transform(training_files.data)
    ValM = vector.transform(val_files.data)
    
    #Testing for variation in C and 'max_iter' to reduce error
    param_grid = {'C': np.logspace(-1,10, 100), 'max_iter': [10,100,1000] }
    GS = GridSearchCV(LogisticRegression(), param_grid)
    GS.fit(TrainM, training_files.target)
    pred = GS.predict(ValM)
    best_C = GS.best_estimator_.C
    best_max_iter = GS.best_estimator_.max_iter
    print("**Grid Search Result: Logistic Regression***")
    print("Best score:",(GS.best_score_*100), "%")
    print("Best estimator of C:",best_C )
    print("Best estimator of max_iter:", best_max_iter )
    print(GS.best_params_)
    print("\n")    
    
GridSarch_NB()
GridSearch_SVM()
GridSearch_LR()