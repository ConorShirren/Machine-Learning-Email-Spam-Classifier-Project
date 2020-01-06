#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:11:46 2018

@author: conorshirren
"""

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
 
    
    
    
    
val_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/validation',description = 'validation emails')
 
    
    
    
    
def classifier_NB():
    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training', 
                          description = ' training emails')

    vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english",
                              strip_accents="ascii",max_df=0.99,min_df=0.01)

    TrainM = vector.fit_transform(training_files.data)
    ValM = vector.transform(val_files.data)
    TrainBOG = TrainM.toarray()
    ValBOG = ValM.toarray()
    
    print("\n----- Naive Bayes Classification Model -----\n")
    nb = MultinomialNB()
    nb.fit(TrainBOG,training_files.target)
    predictNB = nb.predict(ValBOG) 
    NB_accuracy = accuracy_score(val_files.target,predictNB)
    print("Accuracy:    "+ str(round((NB_accuracy*100),3))+"%")
    NB_recall = recall_score(val_files.target,predictNB)
    print("Recall:      "+ str(round((NB_recall*100),3))+"%")
    NB_precision = precision_score(val_files.target,predictNB)
    print("Precision:   "+ str(round((NB_precision*100),3))+"%")
    NB_Fscore = f1_score(val_files.target,predictNB)
    print("F-Score:     "+ str(round((NB_Fscore*100),3))+"%")
 
    Confus = confusion_matrix(val_files.target,predictNB)
#   print("\nConfusion Matrix: \n%s\n" %Confus)
    
    ham_classified = Confus[0,0]
    ham_missclassified = Confus[0,1]
    spam_classified = Confus[1,1]
    spam_missclassified = Confus[1,0]
    
    print("Number of Ham emails correctly Classified:   %s" %ham_classified)
    print("Number of Spam emails correctly Classified:  %s" %spam_classified)
    print("Number of Ham emails Missclassified:         %s" %ham_missclassified)
    print("Number of Spam emails Missclassified:        %s" %spam_missclassified)
    
    
    
def classifier_SVM():

    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training', 
                          description = ' training emails')
    
    
    vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english",
                              strip_accents="ascii",max_df=0.99,min_df=0.01)


    TrainM = vector.fit_transform(training_files.data)
    ValM = vector.transform(val_files.data)
    TrainBOG = TrainM.toarray()
    ValBOG = ValM.toarray()

    print("\n----- Support Vector Classification Model -----\n")
    svc = LinearSVC()
    svc.fit(TrainBOG,training_files.target)
    predictSVC = svc.predict(ValBOG) 
    SVC_accuracy = accuracy_score(val_files.target,predictSVC)
    print("Accuracy:    "+ str(round((SVC_accuracy*100),3))+"%")
    SVC_recall = recall_score(val_files.target,predictSVC)
    print("Recall:      "+ str(round((SVC_recall*100),3))+"%")
    SVC_precision = precision_score(val_files.target,predictSVC)
    print("Precision:   "+ str(round((SVC_precision*100),3))+"%")
    SVC_Fscore = f1_score(val_files.target,predictSVC)
    print("F-Score:     "+ str(round((SVC_Fscore*100),3))+"%")
 
    Confus = confusion_matrix(val_files.target,predictSVC)
#    print("\nConfusion Matrix: \n%s\n" %Confus)
    
    ham_classified = Confus[0,0]
    ham_missclassified = Confus[0,1]
    spam_classified = Confus[1,1]
    spam_missclassified = Confus[1,0]
    
    print("Number of Ham emails correctly Classified:   %s" %ham_classified)
    print("Number of Spam emails correctly Classified:  %s" %spam_classified)
    print("Number of Ham emails Missclassified:         %s" %ham_missclassified)
    print("Number of Spam emails Missclassified:        %s" %spam_missclassified)   
    
    
    
def classifier_LR():

    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training', 
                          description = ' training emails')

    vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english",
                              strip_accents="ascii",max_df=0.99,min_df=0.01)


    TrainM = vector.fit_transform(training_files.data)
    ValM = vector.transform(val_files.data)  
    TrainBOG = TrainM.toarray()
    ValBOG = ValM.toarray()

    print("\n----- Logistic Regression Classification Model -----\n")    
    lr = LogisticRegression()
    lr.fit(TrainBOG,training_files.target)
    predictLR = lr.predict(ValBOG) 
    LR_accuracy = accuracy_score(val_files.target,predictLR)
    print("Accuracy:    "+ str(LR_accuracy*100))#round((LR_accuracy*100),3))+"%")
    LR_recall = recall_score(val_files.target,predictLR)
    print("Recall:      "+ str(round((LR_recall*100),3))+"%")
    LR_precision = precision_score(val_files.target,predictLR)
    print("Precision:   "+ str(round((LR_precision*100),3))+"%")
    LR_Fscore = f1_score(val_files.target,predictLR)
    print("F-Score:     "+ str(round((LR_Fscore*100),3))+"%")
 
    Confus = confusion_matrix(val_files.target,predictLR)
#    print("\nConfusion Matrix: \n%s\n" %Confus)
    
    ham_classified = Confus[0,0]
    ham_missclassified = Confus[0,1]
    spam_classified = Confus[1,1]
    spam_missclassified = Confus[1,0]
    
    print("Number of Ham emails correctly Classified:   %s" %ham_classified)
    print("Number of Spam emails correctly Classified:  %s" %spam_classified)
    print("Number of Ham emails Missclassified:         %s" %ham_missclassified)
    print("Number of Spam emails Missclassified:        %s" %spam_missclassified)    
    
      
def classifier_ADA():

    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training', 
                          description = ' training emails')

    vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english",
                              strip_accents="ascii",max_df=0.99,min_df=0.01)

    TrainM = vector.fit_transform(training_files.data)
    ValM = vector.transform(val_files.data)
    TrainBOG = TrainM.toarray()
    ValBOG = ValM.toarray()

    print("\n----- Adaboost Classification Model -----\n")
    ada = AdaBoostClassifier()
    ada.fit(TrainBOG,training_files.target)
    predictAda = ada.predict(ValBOG) 
    Ada_accuracy = accuracy_score(val_files.target,predictAda)
    print("Accuracy:    "+ str(round((Ada_accuracy*100),3))+"%")
    Ada_recall = recall_score(val_files.target,predictAda)
    print("Recall:      "+ str(round((Ada_recall*100),3))+"%")
    Ada_precision = precision_score(val_files.target,predictAda)
    print("Precision:   "+ str(round((Ada_precision*100),3))+"%")
    Ada_Fscore = f1_score(val_files.target,predictAda)
    print("F-Score:     "+ str(round((Ada_Fscore*100),3))+"%")

    Confus = confusion_matrix(val_files.target,predictAda)
#    print("\nConfusion Matrix: \n%s\n" %Confus)
    
    ham_classified = Confus[0,0]
    ham_missclassified = Confus[0,1]
    spam_classified = Confus[1,1]
    spam_missclassified = Confus[1,0]
    
    print("Number of Ham emails correctly Classified:   %s" %ham_classified)
    print("Number of Spam emails correctly Classified:  %s" %spam_classified)
    print("Number of Ham emails Missclassified:         %s" %ham_missclassified)
    print("Number of Spam emails Missclassified:        %s" %spam_missclassified)

    

def Score_NB():
    start = timer()
    classifier_NB()
    end = timer()
    print("Execution Time for Naive Bayes Classifier is: "+ str(round((end-start),3))+"s")
    
def Score_SVM():
    start = timer()
    classifier_SVM()
    end = timer()
    print("Execution Time for Support Vector Classifier is: "+ str(round((end-start),3))+"s")
    
def Score_LR():
    start = timer()
    classifier_LR()
    end = timer()
    print("Execution Time for Logistic Regression Classifier is: "+ str(round((end-start),3))+"s")
    
def Score_ADA():
    start = timer()
    classifier_ADA()
    end = timer()
    print("Execution Time for Ada Boost Classifier is: "+ str(round((end-start),3))+"s")
      
#classifier_ADA()    
#classifier_NB()
#classifier_SVM()
#classifier_LR()

#Score_ADA()
#Score_NB()
#Score_SVM()
Score_LR()
