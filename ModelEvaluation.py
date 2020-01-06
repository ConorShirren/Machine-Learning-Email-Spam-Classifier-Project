#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:26:15 2018

@author: conorshirren
"""

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score 
from sklearn import metrics

test_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/testing',description = 'testing emails')

def classifier_LR():

    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training', 
                          description = ' training emails')

    vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english",
                              strip_accents="ascii",max_df=0.99,min_df=0.01)


    TrainM = vector.fit_transform(training_files.data)
    TestM = vector.transform(test_files.data)  
    TrainBOG = TrainM.toarray()
    ValBOG = TestM.toarray()
    

    print("\n----- Logistic Regression Classification Model: Out of Sample Test -----\n")    
    lr = LogisticRegression() #Default Paremeters work the best
    lr.fit(TrainBOG,training_files.target)
    predictLR = lr.predict(ValBOG) 
    LR_accuracy = accuracy_score(test_files.target,predictLR)
    print("Accuracy:    "+ str(round((LR_accuracy*100),3))+"%")
    LR_recall = recall_score(test_files.target,predictLR)
    print("Recall:      "+ str(round((LR_recall*100),3))+"%")
    LR_precision = precision_score(test_files.target,predictLR)
    print("Precision:   "+ str(round((LR_precision*100),3))+"%")
    LR_Fscore = f1_score(test_files.target,predictLR)
    print("F-Score:     "+ str(round((LR_Fscore*100),3))+"%")
    
    Confus = confusion_matrix(test_files.target,predictLR)
    print("\nConfusion Matrix: \n%s\n" %Confus)
    
    ham_classified = Confus[0,0]
    ham_missclassified = Confus[0,1]
    spam_classified = Confus[1,1]
    spam_missclassified = Confus[1,0]
    
    print("Number of Ham emails correctly Classified:   %s" %ham_classified)
    print("Number of Spam emails correctly Classified:  %s" %spam_classified)
    print("Number of Ham emails Missclassified:         %s" %ham_missclassified)
    print("Number of Spam emails Missclassified:        %s" %spam_missclassified)
    
    y = test_files.target
    lr_scores = lr.predict_proba(TestM)
    true_lr_scores = lr_scores[:,1]
    roc_auc = roc_auc_score(y, true_lr_scores) 
    print("\n\n\nauc score: ", roc_auc)
    fpr, tpr, thresholds = metrics.roc_curve(y, true_lr_scores)
    # ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='magenta',
             lw=lw, label='ROC curve (area = %0.5f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='teal', lw=lw, linestyle='--') 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve (ROC)') 
    plt.legend(loc="lower right")
    plt.show()
    


def Score_LR():
    start = timer()
    classifier_LR()
    end = timer()
    print("\n\nExecution Time for Logistic Regression Classifier is: "+ str(round((end-start),3))+"s")
    


Score_LR()

