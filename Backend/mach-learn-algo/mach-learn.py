# This algorithm currently reads data, conducts some training and testing outputing an accuracy level 
#
# Import modules
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read data from csv file
data=pd.read_csv('covid-news.csv')
data.head()
print(data)

# Extract labels from csv file
lb=data.label
lb.head()
print(lb)

# Split data for training and testing (currently 80:20 - train:test)
x_train,x_test,y_train,y_test=train_test_split(data['text'], lb, test_size=0.2, random_state=7)

# TFIDF-Vectorizor - text array converted to TF-IDF matrix
# TF (Term Frequency) - Number of times a work appears in text
# IDF (Inverse Document Frequency) - measure of how significant a term is in the entire data
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

# Passive Agressive Classifier updates loss after each iteration and changes weight vector
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Predictions about testing data
y_pred=pac.predict(tfidf_test)
print(y_pred)

# Calculate accuracy of model over testing data
score=accuracy_score(y_test,y_pred)

print("Accuracy: ",round(score*100,2),"%")

