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
    
clean_ingred = ingred_clean(train["ingredients"][0])

# Get the number of reviews based on the dataframe column size
num_ingredients = train["ingredients"].size

# Initialize an empty list to hold the clean reviews
clean_ingred_all = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange( 0, num_ingredients ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_ingred_all.append( ingred_clean(train["ingredients"][i] ) )
    #clean_ingred_all = clean_ingred_all + ( ingred_clean(train["ingredients"][i] ) )

#Create bag of words vector
from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, set max features to 90% 
    # len(set(clean_ingred_all))
vectorizer = CountVectorizer(analyzer = "word",   \
                            tokenizer = None,    \
                            preprocessor = None, \
                            stop_words = None,   \
                            max_features = 2700)

train_data_features = vectorizer.fit_transform(clean_ingred_all)

train_data_features = train_data_features.toarray()

##
#reference file = explore_Enron_data.py Udacity course
#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words                            
##
