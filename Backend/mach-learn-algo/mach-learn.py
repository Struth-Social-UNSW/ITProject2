# This algorithm currently reads data, conducts some training and testing outputing an accuracy level 
#
# Import modules
import numpy as np
import pandas as pd
import itertools

# Import libraries from scikit learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load and read data from csv file
dataset = pd.read_csv('covid-news.csv')
dataset.head()
print(dataset)

# Extract labels from csv file
lb = dataset.label
lb.head()
print(lb)

# Divide data for training and testing (currently 80:20 - train:test)
x_train, x_test, y_train, y_test = train_test_split(dataset['text'], lb, test_size = 0.2, random_state = 7)

# TFIDF-Vectorizor - text array converted to TF-IDF matrix to define importance of keyword
# TF (Term Frequency) - number of times a word appears in text
# IDF (Inverse Document Frequency) - measure of how significant a work is in the entire data
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Passive Agressive Classifier - is an online learning alogorithm which remains passive for a correct classification and turns aggressive for miscalculations.
# It updates loss after each iteration and changes weight vector
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)

# Predictions about testing data
y_pred = pac.predict(tfidf_test)
print(y_pred)

# Calculate accuracy of model over testing data
score = accuracy_score(y_test, y_pred)

print("Accuracy: ", round(score*100, 2), "%")

