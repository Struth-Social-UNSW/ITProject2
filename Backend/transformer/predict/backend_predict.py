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
    print(vector)

main("If you tested positive for #COVID19 and have no symptoms stay home and away from other people. Learn more about CDCï¿½s recommendations about when you can be around others after COVID-19 infection: https://t.co/z5kkXpqkYb. https://t.co/9PaMy0Rxaf")