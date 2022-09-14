""" dl_model.py: 
    
    This program provides the access to the model and generates a many dimensions vector which models
    the relationship of the text between itself as well as the meaning of the text hollistically
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "12 Aug 22"
__Version__     = 1.0

# importing required libraries
from transformers import pipeline
import tensorflow as tf
import numpy as np
import pandas as pd     # for the formatting/reading of data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys

def dlmodel(input):
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        input: a str containing the body of a Tweet (after being preprocessed)

        ** Returns **
        A many dimensional vector modelling the features of the text, as per the pretraining of the model
    """
    pipe = pipeline("zero-shot-classification", model="digitalepidemiologylab/covid-twitter-bert-v2-mnli")    
    fake_real = ['fake', 'real']
    statement = 'This example is {}.'
    result = pipe(input, fake_real, hypothesis_template=statement, multi_label=True)
    return result

def dlmodelmain():
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        N/A

        ** Returns **
        N/A
    """
    preproc = pd.read_csv('preproc_data.csv')
    dl_dict = {'classifier': [], 'vector': []}
    dl_data = pd.DataFrame(data=dl_dict)
    count = 0
    
    
    
    
    
    