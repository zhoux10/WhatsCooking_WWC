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
import numpy
import scipy
import pandas
import matplotlib
import sklearn
import os
from collections import Counter

os.getcwd()
os.chdir('C:\Users\jbark1967\Documents\KaggleContests/WWC_WhatsCooking')

cooking_data = json.load(open("data/train.json"))


#Cuisine Stats


def list_cuisine_all(data):
    cuisine_list = []
    for name in data:
        cuisine_list.append(name['cuisine'])
    return cuisine_list
    
def list_cuisine(data):  #get list of unique cuisines w/o count
    cuisine_set = set()
    for name in data:
        cuisine_set.add(name['cuisine'])
    return list(cuisine_set)
    
def cuisine_count(data): #counter object of unique cuisines w/ count  
#c.item() to ge list)
    cuisine_list = list_cuisine_all(data)
    c = Counter(cuisine_list)
    return c
    
print '# of recipes, ', len(list_cuisine_all(cooking_data))
print cuisine_count(cooking_data)
print "# unique cuisines, ", len(cuisine_count(cooking_data))
    

#Ingredient stats

def list_ingredients_all(data): #list of all ingredients
    ingredients_list = []
    for name in data:
        ingredient_list = name['ingredients']
        for ingred in ingredient_list:
            ingredients_list.append(ingred)
    return ingredients_list
    
def ingredient_count(data):
    ingredients_all = list_ingredients_all(data)
    c = Counter(ingredients_all)
    return c

def list_ingredients(data): #list of unique ingredients
    ingredients_set = set()
    for name in data:
        ingredient_list = name['ingredients']
        for ingred in ingredient_list:
            ingredients_set.add(ingred)
    return ingredients_set
        
print "number of ingredients, ", len(list_ingredients_all(cooking_data))
print ingredient_count(cooking_data)
print "# of unique ingredients, ", len(list_ingredients(cooking_data))


#develop features

meat = ['chicken', 'turkey', 'pepperoni', 'steak', 'beef', 'fish', 'shrimp', 
        'bacon', 'pork', 'prosciutto', 'sausage', 'anchov', 'pancetta', 
        'mussels', 'salmon', 'meat', 'lamb', 'crawfish', 'scallops', 'catfish',
        'clam', 'prawns', 'ham', 'chuck', 'fillets', 'sirloin', 'oyster', 
        'escarole', 'rib', 'salami', 'veal', 'crab', 'oxtails', 'cod', 'chorizo',
        'tilapia', 'rib', 'kielbasa'] #got to qty 27 appearances.  yes or no

carb = []

spice = []
'''
reference file = explore_Enron_data.py Udacity course
'''
