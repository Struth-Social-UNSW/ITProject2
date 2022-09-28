""" backend.py: 
    
    This program receives the text input from the Twitter tweet and controls its movement through the backend
    for preprocessing and other associated computation tasks.
    
    Imports the preprocmain function from the dl_backend_preproc.py file and the dlmodelmain function from the dl_backend_train.py file.
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "28 Sep 22"
__version__     = 3.0   
__status__      = "Complete"
__notes__       = "This program is intended to be run as the driver program for the training pipeline"

# importing utility scripts from other programs
from dl_preproc_train import * 
from dl_model_train import *    


def main(train, test):
    """ This function controls the flow of the program to the various parts of the backend.
        
        ** Parameters **
        train: name of the csv containing training/testing Tweets from the Kaggle Dataset
        test: name of the csv containing validation Tweets fromn the Kaggle Dataset
    """
    preprocmain(train, test)    # Preprocessing
    dlmodelmain()               # Deep Learning Model

main("preproc_data_trg_test.csv", "preproc_data_eval.csv")