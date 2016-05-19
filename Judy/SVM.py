#!/usr/bin/python

""" 
    
    The dataset has the form:
{
 "id": 24717,
 "cuisine": "indian",
 "ingredients": [
     "tumeric",
     "vegetable stock",
     "tomatoes",
     "garam masala",
     "naan",
     "red lentils",
     "red chili peppers",
     "onions",
     "spinach",
     "sweet potatoes"
 ]
 },
    
"""

import json
import csv
import numpy as np
import scipy
import pandas as pd
import matplotlib
import sklearn
import os
from collections import Counter

os.getcwd()
os.chdir('C:\Users\jbark1967\Documents\KaggleContests/WWC_WhatsCooking')

train = pd.read_json("data/train.json")

#Cuisine Stats

train.describe()
train.shape
train.head()

#dataframe mods:  

#add col w/ ingredient counts

train["ingred_count"] = train["ingredients"].apply(len)

#create bag of words (source https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem.snowball import SnowballStemmer

def ingred_clean(input):
    sublist = []
    for x in input:
        #remove everything but letters:
        letters_only = re.sub("[^a-zA-Z]", " ", x) 
 
        #Convert to lower case, split into individual words
        words = letters_only.lower().split() 
        #In Python, searching a set is much faster than searching
        #a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))                  
        # 
        #Remove stop words
        meaningful_words = [w for w in words if not w in stops]
        #print meaningful_words
        #
        #Join the words back into one string separated by space, 
    # and return the result.
        
        sublist = sublist + meaningful_words    #return( " ".join( meaningful_words ))   
        #print sublist #return meaningful_words
    return(" ".join( sublist))
    
train["cl_ing"]=train["ingredients"].apply(ingred_clean)   
    
#split feature/label and train/test
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

feature = train[["cl_ing", "ingred_count"]]
label = train[["cuisine"]]
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(feature, label, test_size=0.2, random_state=42)

#Create bag of words vector
from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, set max features to 90% 
    # len(set(clean_ingred_all))
vectorizer = CountVectorizer(analyzer = "word",   \
                            tokenizer = None,    \
                            preprocessor = None, \
                            stop_words = None,   \
                            max_features = 50)

features_train_v = vectorizer.fit_transform(features_train['cl_ing'])
features_test_v  = vectorizer.transform(features_test['cl_ing'])

feature_name = vectorizer.get_feature_names()

features_train_v = features_train_v.toarray()
features_test_v = features_test_v.toarray()
#label sets to numpy array
labels_train_f = labels_train.values
labels_test_f = labels_test.values

#==============================================================================
# # add in ingred count. need some help to add ingred_count column...
# features_train_f = np.append(features_train_v,features_train["ingred_count"])
# features_test_f = np.append(features_test_v,features_test['ingred_count'])
# 
#==============================================================================
#apply classifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as class_rep

clf = svm.SVC(gamma = 0.1)
clf.fit(features_train_v, labels_train_f)

pred = clf.predict(features_test_v)

accuracy = accuracy_score(labels_test, pred)
report = class_rep(labels_test, pred)
print "accuracy: ", accuracy
print "report: ", report

print feature_name

##
#reference file = explore_Enron_data.py Udacity course
#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words                            
##
