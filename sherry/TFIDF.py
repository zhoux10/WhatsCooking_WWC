# Based on tutorial from:
# http://radimrehurek.com/data_science_python/

# TODO: Find out percentage of false positives, figure out how to filter out
# TODO: Use Pipeline
# TODO: Use built-in splitter for training/testing?

import matplotlib.pyplot as plt
import csv
import pandas
import json
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve
from sklearn.svm import SVC
import re
from nltk.corpus import stopwords # Import the stop word list

# CHANGE TRAIN_FILE NUMBER TO TRAIN ON DIFFERENT FILE
train_file = 4
train_file_name = "./data/train_%d.json" % train_file
test_file = (train_file + 1) % 4
test_file_name = "./data/train_%d.json" % test_file

with open(train_file_name) as file:
    recipes_json = json.load(file)

recipes_split = []

separator = "; "

# Join the ingredients for each recipe object and put into an array
for row in recipes_json:
    current_ingredient = {
        "id": row["id"],
        "cuisine": row["cuisine"],
        "ingredients": separator.join(row["ingredients"])
    }
    recipes_split.append(current_ingredient)

recipes = pandas.DataFrame(data = recipes_split, columns = ["id", "cuisine", "ingredients"])

# Used to clean up un-split-up strings of ingredients separated by ;
# Removes hyphen, parentheticals, and un-informative words
# TODO: Currently, reduced is not working well
def clean_string(string):
    bad_descriptions = [
                          'low\s?[a-z]+',
                          'nondairy',
                          'unsweetened',
                          'dried',
                          'fresh',
                          'summer',
                          '[0-9]+%',
                          'reduced\s?[a-z]+',
                          'lower\s?[a-z]+',
                          'small',
                          'large',
                          'frozen',
                          'homemade',
                          'canned',
                          'nonfat',
                          'freezedried',
                          "whole wheat",
                          "mild",
                          "vegetarian",
                          'glutenfree',
                          "cooked",
                          "raw"
                        ]
    bad_descriptions = ("\s+" + s + "\s+" for s in bad_descriptions)
    bad_descriptions = "|".join(bad_descriptions)
    string = re.sub(r"\(.*?\)|\s*-\s*", "", string.lower())
    string = re.sub(r"%s" % bad_descriptions, "", string)
    return string

# Clean up the string and then separate into ingredients
# Used as tokenizer for CountVectorizer
def separate_into_ingredients(string):
    cleaned_string = clean_string(string)
    return re.split(r"\s*;\s*|\s+or\s+", cleaned_string)

# Bag of Words for Train dataset
bow_transformer = CountVectorizer(tokenizer=separate_into_ingredients, strip_accents='ascii', lowercase="true").fit(recipes['ingredients'])
# print(bow_transformer.vocabulary_)

# Transform to ingredients for the recipe
recipes_bow = bow_transformer.transform(recipes['ingredients'])

# TFIDF transformer and NB model
tfidf_transformer = TfidfTransformer().fit(recipes_bow)
recipes_tfidf = tfidf_transformer.transform(recipes_bow)
recipe_labeler = MultinomialNB(alpha=0).fit(recipes_tfidf, recipes['cuisine'])

# TEST MODEL
total_recipes = 0
failed_recipes = 0
failed_recipes_list = []
# Load test dataset
with open(test_file_name) as file:
    recipes_test_json = json.load(file)

# FUnction for getting overall prediction
def get_prediction(predictions, probability):
    results = {}
    cuisine_to_idx = {}
    max_values = {
        "name": "",
        "value": 0
    }

    for idx, cuis in enumerate(recipe_labeler.classes_):
        cuisine_to_idx[cuis] = idx

    for idx, pred in enumerate(predictions):
        cuisine_idx = cuisine_to_idx[pred]
        cuisine_prob = probability[idx][cuisine_idx]
        if cuisine_prob >= 0.4:
            try:
                results[pred]
                results[pred] = results[pred] + cuisine_prob
            except Exception as e:
                results[pred] = cuisine_prob
            if results[pred] > max_values["value"]:
                max_values["name"] = pred
                max_values["value"] = results[pred]
    return max_values

# For each recipe in test dataset, predict cuisine using ingredients
for test_recipe in recipes_test_json:
    total_recipes = total_recipes + 1
    test_bow = bow_transformer.transform(separate_into_ingredients(separator.join(test_recipe["ingredients"])))
    test_tfidf = tfidf_transformer.transform(test_bow)
    test_recipe["prediction"] = recipe_labeler.predict(test_tfidf)
    test_recipe["probability"] = recipe_labeler.predict_proba(test_tfidf)
    test_recipe["final_prediction"] = get_prediction(test_recipe["prediction"], test_recipe["probability"])
    if test_recipe["final_prediction"]["name"] != test_recipe["cuisine"]:
        print(test_recipe["cuisine"])
        print(test_recipe["final_prediction"])
        failed_recipes = failed_recipes + 1
        failed_recipes_list.append({
          "id": test_recipe["id"],
          "cuisine": test_recipe["cuisine"],
          "prediction": test_recipe["prediction"],
          "probability": test_recipe["probability"],
          "final_prediction": test_recipe["final_prediction"],
        })

with open("data/results.csv", "a") as file:
    output = csv.writer(file)
    output.writerow([train_file_name, test_file_name, total_recipes, failed_recipes, "Use 0.3 cut off", failed_recipes/total_recipes])

print("Failed: ", failed_recipes)
print("Percentage: ", failed_recipes/total_recipes)
