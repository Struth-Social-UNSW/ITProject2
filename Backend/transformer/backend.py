""" backend.py: 
    
    This program receives the text input from the Twitter tweet and controls its movement through the backend
    for preprocessing and other associated computation tasks
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "13 Aug 22"
__Version__     = 1.0

# importing utility scripts
from preproc import *
from dl_model import *


def main(input):
    """ This function controls the flow of the program to the various parts of the backend.
        
        ** Parameters **
        input: a str containing the body of a Tweet
    """
    preproc = preprocmain(input)
    vector = dlmodelmain(preproc)
    print(json.dumps(vector, indent=4))

main("Take simple daily precautions to help prevent the spread of respiratory illnesses like #COVID19. Learn how to protect yourself from coronavirus (COVID-19): https://t.co/uArGZTrH5L. https://t.co/biZTxtUKyK")
