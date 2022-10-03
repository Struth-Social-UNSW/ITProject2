""" dl_backend_predict.py: 
    
    This program receives the text input from the Twitter tweet and controls its movement through the backend
    for preprocessing and Deep Learning tasks.
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "28 Sep 22"
__version__     = 3.0   
__status__      = "Complete"
__notes__       = "This program is intended to be run as the driver program for the training pipeline"

# importing utility scripts
from dl_preproc_predict import *
from dl_model_predict import *


def main(input):
    """ This function controls the flow of the program to the various parts of the backend.
        
        ** Parameters **
        input: a str containing the body of a Tweet
        
        ** Returns **
        result: a list containing the classification of the text and accuracy
    """
    preproc = preprocmain(input)    # Preprocessing
    result = dlmodelmain(preproc)   # Deep Learning Model
    print(result)                   # prints the results
    return(result)                  # returns the results

main("")    