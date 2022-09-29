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
    """
    preproc = preprocmain(input)
    vector = dlmodelmain(preproc)
    print(vector)

main("india records yet another single-day rise of over 28000 new cases while more than 5.5 lakh individuals have recovered from covid-19. kerala government sets up its first plasma bank in the state following in the steps of delhi and west bengal. #covid19 #coronavirusfacts twitterurl")