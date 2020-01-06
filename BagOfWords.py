#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:40:21 2018

@author: conorshirren
"""
#import os
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import os



def bow_for_ham_emails():

     print("\n\nBag of words for ham emails")
     Ham_training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training',
                                   categories = 'ham', shuffle = False)
     
     Ham_Corpus = Ham_training_files.data
     
     Ham_vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",
                                  stop_words="english",strip_accents="ascii",max_features = 20)
     
     Ham_Matrix = Ham_vector.fit_transform(Ham_Corpus)
     
     Ham_BOW = Ham_Matrix.toarray()
     
     print(Ham_BOW)
     
     print(Ham_vector.vocabulary_)
          
     sum_col = Ham_BOW.sum(axis=0)
     
     plot_top_20_words(Ham_vector.vocabulary_,sum_col,'ham')
     


    

def bow_for_spam_emails():

     print("\n\nBag of words for spam emails")
     Spam_training_files=load_files('/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training',categories = 'spam', shuffle = False)
     
     Spam_Corpus = Spam_training_files.data

#     Spam_vector = TfidfVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english",strip_accents="ascii",max_features = 20)
     Spam_vector = CountVectorizer(encoding='utf-8',decode_error='ignore',analyzer="word",stop_words="english",strip_accents="ascii",max_features = 20)
     
     Spam_Matrix = Spam_vector.fit_transform(Spam_Corpus)
     
     Spam_BOW = Spam_Matrix.toarray()
     
     print(Spam_BOW)
     
     print(Spam_vector.vocabulary_)
     
     sum_col = Spam_BOW.sum(axis=0)
     
     plot_top_20_words(Spam_vector.vocabulary_,sum_col,'spam')
   
    
    

def plot_top_20_words(BoW,freq,a):
    
    y_pos = np.arange(len(BoW))
    
    plt.bar(y_pos,freq, color = 'teal')
    
    plt.xticks(y_pos, BoW, rotation = 'vertical')
    
    plt.title("Frequency of words in %s" %a)
    
    plt.xlabel("Word")
    
    plt.ylabel("Frequency")
    
#    
def get_file_len(a):
    
    len_arr = []
    
    src = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training'
    
    src = os.path.join(src,a)
    
    for file in os.listdir(src): 
            stat=os.stat(os.path.join(src,file))
            
            len_arr.append(stat.st_size)

    return len_arr


def box_plot_len():
    
    ham_len = []
    
    spam_len = []
    
    ham_len = get_file_len('ham')
        
    spam_len = get_file_len('spam')  
    
    plt.title('Email Lengths')
    
    data = [ham_len, spam_len]
    
    plt.boxplot(data,0,'') #remove outliers
    
    plt.xticks([1, 2], ['Ham', 'Spam']) 
    
    plt.ylabel('Number of words')
    
    plt.show()





# Functions   
bow_for_ham_emails()
#bow_for_spam_emails()
#box_plot_len()   
#
