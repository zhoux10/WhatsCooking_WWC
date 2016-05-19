# Based on tutorial from:
# http://radimrehurek.com/data_science_python/

import matplotlib.pyplot as plt
import csv
import pandas
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve
from sklearn.svm import SVC

recipes = pandas.read_csv('./data/ingredients_clean_no_label.csv', sep=',', quoting=csv.QUOTE_NONE, encoding="latin",
                           names=["id", "cuisine", "ingredients", "Ingredient_clean", "type_ingredient"])
# print(recipes.groupby('cuisine').describe())
# print(len(recipes))

bow_transformer = CountVectorizer().fit(recipes['Ingredient_clean'])
# print(len(bow_transformer.vocabulary_))

recipes_bow = bow_transformer.transform(recipes['Ingredient_clean'])
# print('sparse matrix shape:', recipes_bow.shape)
# print('number of non-zeros:', recipes_bow.nnz)
# print('sparsity: %.2f%%' % (100.0 * recipes_bow.nnz / (recipes_bow.shape[0] * recipes_bow.shape[1])))

tfidf_transformer = TfidfTransformer().fit(recipes_bow)
recipes_tfidf = tfidf_transformer.transform(recipes_bow)

# TEST
# recipe4 = recipes['Ingredient_clean'][3]
# bow4 = bow_transformer.transform([recipe4])
# tfidf4 = tfidf_transformer.transform(bow4)
#
# recipe_labeler = MultinomialNB().fit(recipes_tfidf, recipes['cuisine'])
# print('predicted:', recipe_labeler.predict(tfidf4)[0])
# print('expected:', recipes.cuisine[3])
