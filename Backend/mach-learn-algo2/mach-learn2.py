# This algorithm currently reads data, conducts some training and testing outputing an accuracy level 
# Detect if a given news article is fake or real
#
# Import modules
from turtle import update
import numpy as np
import pandas as pd
import pickle

# Import libraries from scikit learn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB


# Load and read data from csv file
dataset = pd.read_csv('kaggle-covid-news.csv')

# Remove any unknown or unlabeled rows
dataset.drop(dataset[dataset['label'] == 'U'].index, inplace = True)
#dataset.drop(dataset[(dataset['label'] != 'TRUE') & (dataset['label'] != 'FALSE')].index, inplace = True)

x = dataset['text']
y = dataset['label']
#print(x)
#print(y)
dataset.head()
#print(dataset)
# Rows and columns
print('Dataset rows & columns: ', dataset.shape)

# Pre-processing

# Check dataset for null values
if(dataset.isnull().any):
    print('Null exists')
print(dataset.isnull().any)

# Divide data for training and testing (currently 80:20 - train:test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

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
#print(y_pred)

# Calculate accuracy of model over testing data
score = accuracy_score(y_test, y_pred)

print("Accuracy: ", round(score*100,2), "%")

# Pipeline utility function to train and transform data to text data
pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words = 'english')), ('nbmodel', MultinomialNB())])
pipeline.fit(x_train, y_train)

# Calculate accuracy
score = pipeline.score(x_test, y_test)
print("Accuracy: ", round(score*100,2), "%")

# Performance evaluation table
pred = pipeline.predict(x_test)
print(classification_report(y_test, pred))

# Confusion matrix
print(confusion_matrix(y_test, pred))

with open('model.pkl', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol = pickle.HIGHEST_PROTOCOL)

with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)


news = input("\nEnter tweet: ")

pred = model.predict([news])
print("The tweet is ", pred[0], "\n\n")



print('Please make your selection: ')
print('1. Test Tweet')
print('2. Exit')
selection = int(input())
while selection != 2:
    news = input("\nEnter tweet: ")
    pred = model.predict([news])
    print("The tweet is ", pred[0], "\n\n")
    print('Please make your selection: ')
    print('1. Test Tweet')
    print('2. Exit')
    selection = int(input())