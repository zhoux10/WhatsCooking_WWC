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
import matplotlib.pyplot as plt
import sklearn
import os
from collections import Counter

train = pd.read_json("../data/train.json")

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
        #print(meaningful_words)
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
                            ngram_range=(1,2), \
                            tokenizer = None,    \
                            preprocessor = None, \
                            max_df = 0.5, \
                            min_df = 1, \
                            stop_words = None,   \
                            max_features = 5000)

stopwords = vectorizer.fit(features_train['cl_ing']).stop_words_

features_train_v = vectorizer.fit_transform(features_train['cl_ing'])
features_test_v  = vectorizer.transform(features_test['cl_ing'])

feature_name = vectorizer.get_feature_names()

features_train_v = features_train_v.toarray()
#features_train.loc[:,'ingred_count'] = pd.Series(features_train.loc[:,['ingred_count']])
features_test_v = features_test_v.toarray()

ing_arr=np.array(features_test[['ingred_count']].values)
ing_arr_train = np.array(features_train[['ingred_count']])

features_test_f = np.append(features_test_v, ing_arr, axis = 1)
features_train_f = np.append(features_train_v, ing_arr_train, axis = 1)
#label sets to numpy array
#labels_train_f = np.arrray(labels_train[['cuisine']].values)
labels_train_f = labels_train.as_matrix()
labels_test_f = labels_test.as_matrix()

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

clf = svm.SVC(kernel = 'linear')
clf.fit(features_train_f, labels_train_f)

pred = clf.predict(features_test_f)

accuracy = accuracy_score(labels_test, pred)
report = class_rep(labels_test_f, pred)
print "accuracy: ", accuracy
print "report: ", report
report = class_rep(labels_test, pred)
print("accuracy: ", accuracy)
print("report: ", report)

print(feature_name)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels_test_f))
    #plt.xticks(tick_marks, labels_test_f, rotation=45)
    #plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(labels_test_f, pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure(figsize = (12,8))
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure(figsize=(20,8))
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()
#==============================================================================
#==============================================================================
# # clf = svm.SVC(gamma = 0.1)
# accuracy:  0.762916404777

#clf = svm.SVC()
#accuracy:  0.552482715273

#clf = svm.SVC(kernel = 'linear')
#accuracy:  0.770710245129



#==============================================================================
#==============================================================================
##
#reference file = explore_Enron_data.py Udacity course
#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
##
