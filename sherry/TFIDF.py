# Based on tutorial from:
# http://radimrehurek.com/data_science_python/

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


train_file = 0
train_file_name = "./data/train_%d.json" % train_file
test_file = (train_file + 1) % 4
test_file_name = "./data/train_%d.json" % test_file

with open(train_file_name) as file:
    recipes_json = json.load(file)

recipes_split = []

for row in recipes_json:
    current_ingredient = {
        "id": row["id"],
        "cuisine": row["cuisine"],
        "ingredients": "; ".join(row["ingredients"])
    }
    recipes_split.append(current_ingredient)

recipes = pandas.DataFrame(data = recipes_split, columns = ["id", "cuisine", "ingredients"])

# recipes = pandas.read_csv('./data/train_0.csv', sep=',', quoting=csv.QUOTE_NONE, encoding="latin",
#                            names=["id", "cuisine", "ingredients"])
# print(recipes.groupby('cuisine').describe())
# print(len(recipes))

def clean_string(string):
    bad_descriptions = [
                          'low\s*[a-z]*?',
                          'nondairy',
                          'unsweetened',
                          'dried',
                          'fresh',
                          'summer',
                          '[0-9]+%',
                          'reduced\s*[a-z]*?',
                          'lower\s*[a-z]*?',
                          'small',
                          'large',
                          'frozen',
                          'homemade',
                          'canned',
                          'nonfat',
                          'freeze-dried',
                          "whole wheat",
                          "mild",
                          "vegetarian"
                        ]
    bad_descriptions = ("\s" + s + "\s+" for s in bad_descriptions)
    bad_descriptions = "|".join(bad_descriptions)
    string = re.sub(r"\(.*?\)|\s*-\s*", "", string.lower())
    string = re.sub(r"%s" % bad_descriptions, " ", string)
    return string

def separate_into_ingredients(string):
    return re.split(r"\s*;\s*|\s+or\s+", clean_string(string))

bow_transformer = CountVectorizer(tokenizer=separate_into_ingredients, strip_accents='ascii', lowercase="true").fit(recipes['ingredients'])
print(bow_transformer.vocabulary_)

recipes_bow = bow_transformer.transform(recipes['ingredients'])
# print('sparse matrix shape:', recipes_bow.shape)
# print('number of non-zeros:', recipes_bow.nnz)
# print('sparsity: %.2f%%' % (100.0 * recipes_bow.nnz / (recipes_bow.shape[0] * recipes_bow.shape[1])))

tfidf_transformer = TfidfTransformer(smooth_idf=False).fit(recipes_bow)
recipes_tfidf = tfidf_transformer.transform(recipes_bow)
recipe_labeler = MultinomialNB().fit(recipes_tfidf, recipes['cuisine'])

# TEST MODEL
total_recipes = 0
failed_recipes = 0
failed_recipes_list = []
with open(train_file_name) as file:
    recipes_train_json = json.load(file)

for test_recipe in recipes_train_json:
    total_recipes = total_recipes + 1
    test_bow = bow_transformer.transform(map(clean_string, test_recipe["ingredients"]))
    test_tfidf = tfidf_transformer.transform(test_bow)
    test_recipe["prediction"] = recipe_labeler.predict(test_tfidf)[0]
    if test_recipe["prediction"] != test_recipe["cuisine"]:
        failed_recipes = failed_recipes + 1
        failed_recipes_list.append({
          "id": test_recipe["id"],
          "cuisine": test_recipe["cuisine"],
          "prediction": test_recipe["prediction"],
        })

failed_recipe_pandas = pandas.DataFrame(data = failed_recipes_list, columns = ["id", "cuisine", "prediction"])
with open("data/results.csv", "a") as file:
    output = csv.writer(file)
    output.writerow([train_file_name, test_file_name, total_recipes, failed_recipes, failed_recipe_pandas.groupby('cuisine').describe(), failed_recipe_pandas.groupby('prediction').describe(), "Try sublinear_tf=True"])

print("Failed: ", failed_recipes)
# # TEST
# recipe4 = recipes['ingredients'][3]
# bow4 = bow_transformer.transform([recipe4])
# tfidf4 = tfidf_transformer.transform(bow4)
#
# print('predicted:', recipe_labeler.predict(tfidf4)[0])
# print('expected:', recipes.cuisine[3])
