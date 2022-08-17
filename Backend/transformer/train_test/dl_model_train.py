""" dl_model.py: 
    
    This program provides the access to the model and generates a many dimensions vector which models
    the relationship of the text between itself as well as the meaning of the text hollistically
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "12 Aug 22"
__Version__     = 1.0

# importing required libraries
from sre_parse import Tokenizer
from transformers import pipeline
import tensorflow as tf
import json
import numpy as np
import pandas as pd     # for the formatting/reading of data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

def dlmodel(input):
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        input: a str containing the body of a Tweet (after being preprocessed)

        ** Returns **
        A many dimensional vector modelling the features of the text, as per the pretraining of the model
    """
    pipe = pipeline(task='feature-extraction', model='digitalepidemiologylab/covid-twitter-bert-v2')
    out = pipe(input)
    array = out[0][0]
    return array

def mltraining(features, labels):
    X = features
    y = labels
    print("Beginnning Fit Task")
    clf.fit(X, y)
    
def mltesting(feature, label):
    result = clf.predict(feature)
    if result == label:
        print('Success')
    else:
        print("Failed")

def dlmodelmain():
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        N/A

        ** Returns **
        N/A
    """
    preproc = pd.read_csv('./preproc_data.csv')
    dl_dict = {'classifier': [], 'vector': []}
    dl_data = pd.DataFrame(data=dl_dict)
    count = 0
    
    listfeaturesx = []
    featuresx = np.array([])
    listfeaturesy = []
    featuresy = np.array([])
    
    for index in preproc.index:
        print('Text '+str(count)+' starting')
        vector = dlmodel(str(preproc['text'][index]))
        # print(vector)
        listfeaturesx.insert(count, list(vector))
        print(listfeaturesx)
        vectorised = {'classifier': [preproc['classifier'][index]], 'vector': [vector]}
        dl_add = pd.DataFrame(vectorised)
        dl_data = pd.concat([dl_data, dl_add], sort=False)
        count = count+1
    
    
    
    
    
    