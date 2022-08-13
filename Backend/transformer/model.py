""" model.py: 
    
    This program provides the access to the model and generates a many dimensions vector which models
    the relationship of the text between itself as well as the meaning of the text hollistically
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "12 Aug 22"
__Version__     = 1.0

# importing required libraries
from transformers import pipeline
import json

def modelmain(input):
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        input: a str containing the body of a Tweet (after being preprocessed)

        ** Returns **
        A many dimensional vector modelling the features of the text, as per the pretraining of the model
    """
    pipe = pipeline(task='feature-extraction', model='digitalepidemiologylab/covid-twitter-bert-v2')
    out = pipe(input)
    print(json.dumps(out, indent=4))

modelmain("This is a test of the functionality")