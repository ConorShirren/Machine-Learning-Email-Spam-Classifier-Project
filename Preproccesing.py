#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 23:34:16 2018

@author: conorshirren
"""

import os
import shutil
from random import shuffle
import fnmatch


def ham_emails():
    i = 0
    j = 1
    print("Get ham emails")
    # Set up source folder to retrieved ham emails from 
    src = '/Users/conorshirren/Desktop/EE514Assignment/Data/enron/pre/'
    print("Get ham emails from: "+ str(src))
    # Set up destination folder for all ham emails 
    dst = '/Users/conorshirren/Desktop/EE514Assignment/prepro/ham_emails'
    print("Copy ham emails to: "+ str(dst))
    # Create destination folder if doesn't exist
    if not os.path.exists(dst):
        os.mkdir(dst)
        print("Directory " , dst ,  " Created ")
    else:    
        print("Directory " , dst ,  " already exists")
    # Iterate through each enron folder to retrieve all files (1-6)    
    for folder in os.listdir(src):
        # Join the string 'enron' + j to end of src patch
        src_enron = os.path.join(src,'enron'+str(j)+'/ham')
        print(src_enron)
        for file in os.listdir(src_enron):
            # Copy all files from scr_enron to destination folder
            shutil.copyfile((os.path.join(src_enron,file)),(os.path.join(dst,file)))
            i = i + 1
        j = j +1
    print("Number of Ham Files: %s" % i)    # Represents the total number of ham files
    print("Number of Folders: %s" % j)      # Represents the total number of enron folder that contained ham files

def spam_emails():
    i = 0
    j = 1
    print("Get spam emails")
    # Set up source folder to retrieved ham emails from 
    src = '/Users/conorshirren/Desktop/EE514Assignment/Data/enron/pre/'
    print("Get spam emails from: "+ str(src))
    # Set up destination folder for all ham emails 
    dst = '/Users/conorshirren/Desktop/EE514Assignment/prepro/spam_emails/'
    print("Copy spam emails to: "+ str(dst))
    # Create destination folder if doesn't exist
    if not os.path.exists(dst):
        os.mkdir(dst)
        print("Directory " , dst ,  " Created ")
    else:    
        print("Directory " , dst ,  " already exists")
    # Iterate through each enron folder to retrieve all files (1-6)    
    for folder in os.listdir(src):
        # Join the string 'enron' + j to end of src patch
        src_enron = os.path.join(src,'enron'+str(j)+'/spam')
        print(src_enron)
        for file in os.listdir(src_enron):
            # Copy all files from scr_enron to destination folder
            shutil.copyfile((os.path.join(src_enron,file)),(os.path.join(dst,file)))
            i = i + 1
        j = j + 1
    print("Number of Spam Files: %s" % i)    # Represents the total number of spam files
    print("Number of Folders: %s" % j)      # Represents the total number of enron folder that contained ham files


def all_emails():
    print ("Moving all spam and ham emails to a single folder")
    src = '/Users/conorshirren/Desktop/EE514Assignment/prepro'
    print("Retrieving all emails from: "+ str(src))
    # Define distenation path for all emails
    dst = '/Users/conorshirren/Desktop/EE514Assignment/prepro/all_emails'
    print("Copy all emails to: "+ str(dst))
    # Define hams and spam src locations
    ham_emails = '/Users/conorshirren/Desktop/EE514Assignment/prepro/ham_emails/'
    spam_emails = '/Users/conorshirren/Desktop/EE514Assignment/prepro/spam_emails/'
    # Create destination folder if doesn't exist
    if not os.path.exists(dst):
        os.mkdir(dst)
        print("Directory " , dst ,  " Created ")
    else:    
        print("Directory " , dst ,  " already exists")
    # Copy all ham emails to destination 
    for file in os.listdir(ham_emails):
        shutil.copyfile((os.path.join(ham_emails, file)),(os.path.join(dst, file)))
    # Copy all spam emails to destination    
    for file in os.listdir(spam_emails):
        shutil.copyfile((os.path.join(spam_emails, file)),(os.path.join(dst, file)))
 

def train_validate_test():

   print("Split files into training, validation & testing")
   # Define source path for files
   src = '/Users/conorshirren/Desktop/EE514Assignment/prepro/all_emails'
   # Define distination path for files 
   dst1 = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data'
   if not os.path.exists(dst1):
        os.mkdir(dst1)
        print("Directory " , dst1 ,  " Created ")
   else:    
        print("Directory " , dst1 ,  " already exists")
   # Define directory for Training set     
   dst = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training'
   # Create destination directory if doesnt exist
   if not os.path.exists(dst):
        os.mkdir(dst)
        print("Directory " , dst ,  " Created ")
   else:    
        print("Directory " , dst ,  " already exists")  
   # Create an array to store all the files 
   file_arr=[]
   for file in os.listdir(src):
       file_arr.append(file)
   # Shuffle the array to mixed the spam and ham emails    
   shuffle(file_arr)

   # Partition emials into training, validation, and testing sets 
   path,dirs, files = os.walk(src).__next__()
   num_of_files = len(file_arr)
   num_of_validat_files = int(num_of_files*0.2)     # 20% validation
   num_of_test_files = int(num_of_files*0.3)        # 30% testing
   num_of_train_files = int(num_of_files*0.5)       # 50% training
   print("Number of files:                  %s" % num_of_files)
   print("Number of files for Training:     %s" %num_of_train_files)
   print("Number of files for Validation:   %s" %num_of_validat_files)
   print("Number of files for Testing:      %s" %num_of_test_files)
   print("Copying to validation")
   dst = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training'
   for s in file_arr:
      # start copying files to validation dst 
      shutil.copy((os.path.join(src, s)),(os.path.join(dst, s)))
      num_of_files = num_of_files - 1
      # if 20% of total emails is reached, chaged dst to testing
      if num_of_files == num_of_validat_files:
          print("Copying to testing")
          dst = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/validation'
          # create directory if doesnt exist
          if not os.path.exists(dst):
              os.mkdir(dst)
              print("Directory " , dst ,  " Created ")
          else:    
              print("Directory " , dst ,  " already exists")
      # if 50% of total emails is reached, change dst training
      if num_of_files == num_of_train_files:
          print("Copying to training")
          dst = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/testing'    
          # create directory if doesnt exist    
          if not os.path.exists(dst):
              os.mkdir(dst)   
              print("Directory " , dst ,  " Created ")
          else:    
              print("Directory " , dst ,  " already exists")
                  
                  
   print("Files Split into Training (& Validation) or Testing")
   # end of train_validate_test()

def split_training_ham_spam():      
   
    print("\nSplit files in training to ham and spam")
    src = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training' 

   
    for file in  os.listdir(src):
        
        dst1 = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training/ham'
        if not os.path.exists(dst1):
              os.mkdir(dst1)   
              print("\nDirectory " , dst1 ,  " Created ")
        else:    
              print("\nDirectory " , dst1 ,  " already exists")
              
        dst2 = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training/spam'
        if not os.path.exists(dst2):
              os.mkdir(dst2)   
              print("\nDirectory " , dst2 ,  " Created ")
        else:    
              print("\nDirectory " , dst2 ,  " already exists")
        
        
        print("\nSource: %s" %src)
        
        src = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training' 
        dst1 = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training/ham'
        dst2 = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/training/spam'
        if fnmatch.fnmatch(file,'*.ham.*'):
            print("\nDestination: %s" %dst1)
            shutil.move((os.path.join(src, file)),(os.path.join(dst1, file)))
        if fnmatch.fnmatch(file,'*.spam.*'):
            print("\nDestination: %s" %dst2)
            shutil.move((os.path.join(src, file)),(os.path.join(dst2, file)))
   
def split_validation_ham_spam():      
   
    print("\nSplit files in validation to ham and spam")
    src = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/validation' 

   
    for file in  os.listdir(src):
        
        dst1 = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/validation/ham'
        if not os.path.exists(dst1):
              os.mkdir(dst1)   
              print("\nDirectory " , dst1 ,  " Created ")
        else:    
              print("\nDirectory " , dst1 ,  " already exists")
              
        dst2 = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/validation/spam'
        if not os.path.exists(dst2):
              os.mkdir(dst2)   
              print("\nDirectory " , dst2 ,  " Created ")
        else:    
              print("\nDirectory " , dst2 ,  " already exists")
        
        
        print("\nSource: %s" %src)
        
        
        if fnmatch.fnmatch(file,'*.ham.*'):
            print("\nDestination: %s" %dst1)
            shutil.move((os.path.join(src, file)),(os.path.join(dst1, file)))
        if fnmatch.fnmatch(file,'*.spam.*'):
            print("\nDestination: %s" %dst2)
            shutil.move((os.path.join(src, file)),(os.path.join(dst2, file)))
   
def split_test_ham_spam():      
   
    print("\nSplit files in test to ham and spam")
    src = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/testing' 

   
    for file in  os.listdir(src):
        
        dst1 = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/testing/ham'
        if not os.path.exists(dst1):
              os.mkdir(dst1)   
              print("\nDirectory " , dst1 ,  " Created ")
        else:    
              print("\nDirectory " , dst1 ,  " already exists")
              
        dst2 = '/Users/conorshirren/Desktop/EE514Assignment/prepro/sk_data/testing/spam'
        if not os.path.exists(dst2):
              os.mkdir(dst2)   
              print("\nDirectory " , dst2 ,  " Created ")
        else:    
              print("\nDirectory " , dst2 ,  " already exists")
        
        
        print("\nSource: %s" %src)
        
        
        if fnmatch.fnmatch(file,'*.ham.*'):
            print("\nDestination: %s" %dst1)
            shutil.move((os.path.join(src, file)),(os.path.join(dst1, file)))
        if fnmatch.fnmatch(file,'*.spam.*'):
            print("\nDestination: %s" %dst2)
            shutil.move((os.path.join(src, file)),(os.path.join(dst2, file)))
   


# Calling define functions
#ham_emails()
#spam_emails()
#all_emails()
#train_validate_test()
#split_training_ham_spam()
#split_validation_ham_spam()
#split_test_ham_spam()
