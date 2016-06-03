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
# train_file_name = "./data/train_%d.json" % train_file
train_file_name = "./data/train_no_var.json"
test_file = (train_file + 1) % 4
# test_file_name = "./data/train_%d.json" % test_file
test_file_name = "./data/test_no_var.json"

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
recipe_labeler = MultinomialNB(alpha=0, fit_prior=True).fit(recipes_tfidf, recipes['cuisine'])

# TEST MODEL
total_recipes = 0
failed_recipes = 0
failed_types = {}
success_types = {}
failed_recipes_list = []
# Load test dataset
with open(test_file_name) as file:
    recipes_test_json = json.load(file)

# FUnction for getting overall prediction
def get_prediction(predictions, probability):
    results = {}
    all_results = {}
    cuisine_to_idx = {}
    max_values = {
        "name": "",
        "value": 0
    }
    blank_predictions_for_train_dataset = [
        {
        'actual_name':'french',
        'number':122
        },
        {
        'actual_name':'southern_us',
        'number':79
        },
        {
        'actual_name':'british',
        'number':69
        },
        {
        'actual_name':'irish',
        'number':57
        },
        {
        'actual_name':'italian',
        'number':40
        },
       {
          'actual_name':'russian',
          'number':38
       },
       {
       'actual_name':'filipino',
       'number':35
       },
       {
       'actual_name':'mexican',
       'number':30
       },
       {
       'actual_name':'chinese',
       'number':25
       },
       {
       'actual_name':'indian',
       'number':24
       },
       {
       'actual_name':'cajun_creole',
       'number':23
       },
       {
       'actual_name':'spanish',
       'number':23
       },
       {
       'actual_name':'japanese',
       'number':21
       },
       {
       'actual_name':'jamaican',
       'number':22
       },
       {
       'actual_name':'brazilian',
       'number':21
       },
       {
          'actual_name':'vietnamese',
          'number':20
       },
       {
       'actual_name':'korean',
       'number':17
       },
       {
          'actual_name':'moroccan',
          'number':15
       },
       {
       'actual_name':'greek',
       'number':9
       },
       {
          'actual_name':'thai',
          'number':3
       },
    ]
    predictions_if_failed = [
        'french','southern_us','british','irish','italian','russian','filipino',
        'mexican','chinese','indian','cajun_creole','spanish','japanese',
        'jamaican','brazilian','vietnamese','korean','moroccan','greek','thai'
    ]

    for idx, cuis in enumerate(recipe_labeler.classes_):
        cuisine_to_idx[cuis] = idx

    for idx, pred in enumerate(predictions):
        cuisine_idx = cuisine_to_idx[pred]
        cuisine_prob = probability[idx][cuisine_idx]
        if cuisine_prob >= 0.35:
            try:
                results[pred]
                results[pred] = results[pred] + cuisine_prob**2
            except Exception as e:
                results[pred] = cuisine_prob**2
            if results[pred] > max_values["value"]:
                max_values["name"] = pred
                max_values["value"] = results[pred]
        else:
            all_results[pred] = True

    if max_values["name"] == "":
        prediction = ""
        for p in predictions_if_failed:
            if prediction == "":
                try:
                    all_results[p]
                    prediction = p
                except Exception as e:
                    pass
        max_values["name"] = prediction
    return max_values

# For each recipe in test dataset, predict cuisine using ingredients
for test_recipe in recipes_test_json:
    total_recipes = total_recipes + 1
    test_bow = bow_transformer.transform(separate_into_ingredients(separator.join(test_recipe["ingredients"])))
    test_tfidf = tfidf_transformer.transform(test_bow)
    test_recipe["prediction"] = recipe_labeler.predict(test_tfidf)
    test_recipe["probability"] = recipe_labeler.predict_proba(test_tfidf)
    test_recipe["final_prediction"] = get_prediction(test_recipe["prediction"], test_recipe["probability"])
    prediction_name = test_recipe["final_prediction"]["name"]

    # WRITE row
    with open("data/nb_results.csv", "a") as file:
        output = csv.writer(file)
        output.writerow(["id", "cuisine"])
        output.writerow([test_recipe["id"], prediction_name])

# TEST ROW
#     actual_name = test_recipe["cuisine"]
#     # if prediction_name == "":
#     #     prediction_name = "italian"
#     if prediction_name != actual_name:
#         # print(actual_name)
#         # print(test_recipe["final_prediction"])
#         failed_recipes = failed_recipes + 1
#         try:
#             failed_types[actual_name][prediction_name] = failed_types[actual_name][prediction_name] + 1
#         except Exception as e:
#             try:
#                 failed_types[actual_name][prediction_name] = 1
#             except Exception as e:
#                 failed_types[actual_name] = {}
#                 failed_types[actual_name][prediction_name] = 1
#
#         failed_recipes_list.append({
#           "id": test_recipe["id"],
#           "cuisine": actual_name,
#           "prediction": test_recipe["prediction"],
#           "probability": test_recipe["probability"],
#           "final_prediction": prediction_name,
#         })
#     else:
#         try:
#             success_types[test_recipe["cuisine"]] = success_types[test_recipe["cuisine"]] + 1
#         except Exception as e:
#             success_types[test_recipe["cuisine"]] = 1
#
# with open("data/results.csv", "a") as file:
#     output = csv.writer(file)
#     output.writerow([train_file_name, test_file_name, total_recipes, failed_recipes, "Try with smart fills", failed_recipes/total_recipes, success_types, failed_types])
#
# blank_values = []
# # for cuisine, failed in failed_types.items():
# #     blank_values.append({
# #         "actual_name": cuisine,
# #         "number": failed[""]
# #     })
#
# print("Failed: ", failed_recipes)
# print("Percentage: ", failed_recipes/total_recipes)
# print("Failed JSON: ", failed_types)
# print("Blanks: ", blank_values)
