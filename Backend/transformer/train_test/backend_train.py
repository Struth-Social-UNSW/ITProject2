""" backend.py: 
    
    This program receives the text input from the Twitter tweet and controls its movement through the backend
    for preprocessing and other associated computation tasks
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "13 Aug 22"
__Version__     = 1.0

# importing utility scripts
from preproc_train import *
from dl_model_train import *
import pandas as pd


def main(train, test):
    """ This function controls the flow of the program to the various parts of the backend.
        
        ** Parameters **
        train: tsv containing training Tweets from the Kaggle Dataset
        test: tsv containing testing/validation Tweets fromn the Kaggle Dataset
    """
    preproc_train = preprocmain(train)
    vector = dlmodelmain(preproc)
    print(vector)

training = pd.read_csv('../trg_data/train.tsv',sep='\t')
testing = pd.read_csv('../trg_data/dev.tsv',sep='\t')

main(training, testing)