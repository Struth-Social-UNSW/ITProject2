""" backend.py: 
    
    This program receives the text input from the Twitter tweet and controls its movement through the backend
    for preprocessing and other associated computation tasks
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "13 Aug 22"
__Version__     = 1.0

# importing utility scripts
from os import getcwd
from preproc_train import *
from dl_model_train import *
import pandas as pd
import os


def main(train, test):
    """ This function controls the flow of the program to the various parts of the backend.
        
        ** Parameters **
        train: name of the tsv containing training Tweets from the Kaggle Dataset
        test: name of the tsv containing testing/validation Tweets fromn the Kaggle Dataset
    """
    preproc_train = preprocmain(train, test)
    vector = dlmodelmain(preproc)
    print(vector)

# testing = pd.read_csv('./trg_data/dev.tsv',sep='\t')

main("train.csv", "dev.csv")