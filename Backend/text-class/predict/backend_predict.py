""" backend.py: 
    
    This program receives the text input from the Twitter tweet and controls its movement through the backend
    for preprocessing and other associated computation tasks
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "13 Aug 22"
__Version__     = 1.0

# importing utility scripts
from preproc_predict import *
from dl_model_predict import *


def main(input):
    """ This function controls the flow of the program to the various parts of the backend.
        
        ** Parameters **
        input: a str containing the body of a Tweet
        
        ** Returns **
        result: a list containing the classification of the text and accuracy
    """
    preproc = preprocmain(input)
    result = dlmodelmain(preproc)
    print(result)
    return(result)

main("The coronavirus pandemic has forced Florida racing tracks to close, and 1,500 greyhounds are now looking for homes")