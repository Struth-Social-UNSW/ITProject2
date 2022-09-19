""" backend.py: 
    
    This program receives the text input from the Twitter tweet and controls its movement through the backend
    for preprocessing and other associated computation tasks
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "19 Sep 22"
__Version__     = 2.0

# importing utility scripts
from preproc_train import *
from dl_model_train import *


def main(train, test):
    """ This function controls the flow of the program to the various parts of the backend.
        
        ** Parameters **
        train: name of the csv containing training Tweets from the Kaggle Dataset
        test: name of the csv containing testing/validation Tweets fromn the Kaggle Dataset
    """
    preprocmain(train, test)
    dlmodelmain()

main("train.csv", "dev.csv")