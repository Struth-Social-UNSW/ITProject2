# Import modules
import numpy as np
import pandas as pd
import itertools

from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read data from csv file
data=pd.read_csv('general-news.csv')
data.head()

# Extract labels from csv file and print first five labels
lb=df.label
lb.head()

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

# Calculate accuracy of model over testing data
score=accuracy_score(y_test,y_pred)

print("Accuracy: ",round(score*100,2),"%")
