#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:11:46 2018

@author: conorshirren
"""

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer


val_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/validation',description = 'validation emails')


def classifier_NB_Count():
    val_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/validation',
                         description = 'validation emails')
    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training', 
                          description = ' training emails')

    print("\n\nUsing a max_df value of 99%, and a min_df value of 1%")
    vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",
                             stop_words="english", strip_accents="ascii",max_df=0.99,min_df=0.01)   
    TrainM = vector.fit_transform(training_files.data)
    ValM = vector.transform(val_files.data)
    TrainBOG = TrainM.toarray()
    ValBOG = ValM.toarray()

    print("\n----- Classifier NB: Count -----\n")
    #***target****
    #**************Classifier***************
    nb = MultinomialNB()
    nb.fit(TrainBOG,training_files.target)
    predictNB = nb.predict(ValBOG) 
    NB_accuracy = accuracy_score(val_files.target,predictNB)
    print("Accuracy of Naive Bayes Classification Model using CountVectorizer (max_df = 0.99, min_df = 0.01) is : "+ str(round((NB_accuracy*100),3))+"%")
    
def classifier_NB_TFIDF():
    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training', description = ' training emails')
    
    
    vector = TfidfVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english", strip_accents="ascii",max_df=0.99,min_df=0.01)   
  

    TrainM = vector.fit_transform(training_files.data)
    ValM = vector.transform(val_files.data)
    
    TrainBOG = TrainM.toarray()
    ValBOG = ValM.toarray()


    print("\n----- Classifier NB: TF-IDF -----\n")
    
    #***target****
    #**************Classifier***************
    nb = MultinomialNB()
    nb.fit(TrainBOG,training_files.target)
    predictNB = nb.predict(ValBOG) 
    NB_accuracy = accuracy_score(val_files.target,predictNB)
    print("Accuracy of Naive Bayes Classification Model using TfidfVectorizer (max_df = 0.99, min_df = 0.01) is : "+ str(round((NB_accuracy*100),3))+"%")

def Score_NB_Count():
    start = timer()
    classifier_NB_Count()
    end = timer()
    print("Execution Time for Naive Bayes Classifier using CountVectorizer is: "+ str(round((end-start),3))+"s")
    
def Score_NB_TFIDF():
    start = timer()
    classifier_NB_TFIDF()
    end = timer()
    print("Execution Time for Naive Bayes Classifier using TFIDFVectorizer is: "+ str(round((end-start),3))+"s")
    
    
    
    
    
def classifier_NB_Count2():
    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training', 
                          description = ' training emails')

    
    vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english", strip_accents="ascii")
    
  
    TrainM = vector.fit_transform(training_files.data)
    ValM = vector.transform(val_files.data)
    
    TrainBOG = TrainM.toarray()
    ValBOG = ValM.toarray()

    print("\n----- Classifier NB: Count -----\n")
    #***target****
    #**************Classifier***************
    nb = MultinomialNB()

    nb.fit(TrainBOG,training_files.target)
    predictNB = nb.predict(ValBOG) 
    NB_accuracy = accuracy_score(val_files.target,predictNB)
    print("Accuracy of Naive Bayes Classification Model using CountVectorizer (without max_df & min_df) is : "+ str(round((NB_accuracy*100),3))+"%")
    
def classifier_NB_TFIDF2():
    training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training', description = ' training emails')
    
    
    vector = TfidfVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english", strip_accents="ascii")   
  

    TrainM = vector.fit_transform(training_files.data)
    ValM = vector.transform(val_files.data)
    
    TrainBOG = TrainM.toarray()
    ValBOG = ValM.toarray()


    print("\n----- Classifier NB: TF-IDF -----\n")
    
    #***target****
    #**************Classifier***************
    nb = MultinomialNB()
    nb.fit(TrainBOG,training_files.target)
    predictNB = nb.predict(ValBOG) 
    NB_accuracy = accuracy_score(val_files.target,predictNB)
    print("Accuracy of Naive Bayes Classification Model using TfidfVectorizer (without max_df & min_df) is : "+ str(round((NB_accuracy*100),3))+"%")

def Score_NB_Count2():
    start = timer()
    classifier_NB_Count2()
    end = timer()
    print("Execution Time for Naive Bayes Classifier using CountVectorizer (without max_df & min_df) is: "+ str(round((end-start),3))+"s")
    
def Score_NB_TFIDF2():
    start = timer()
    classifier_NB_TFIDF2()
    end = timer()
    print("Execution Time for Naive Bayes Classifier using TFIDFVectorizer (without max_df & min_df) is: "+ str(round((end-start),3))+"s")



#classifier_NB_Count()
#classifier_NB_TFIDF()
    
    
Score_NB_Count()
Score_NB_TFIDF()
Score_NB_Count2()
Score_NB_TFIDF2()