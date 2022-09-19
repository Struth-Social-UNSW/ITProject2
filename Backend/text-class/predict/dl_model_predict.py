""" dl_model.py: 
    
    This program provides the access to the model and generates a many dimensions vector which models
    the relationship of the text between itself as well as the meaning of the text hollistically
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "12 Aug 22"
__Version__     = 1.0

# importing required libraries
from transformers import pipeline

def dlmodelmain(input):
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        input: a str containing the body of a Tweet (after being preprocessed)

        ** Returns **
        label: a str (either fake or real) based on the model's classification
        score: the score (certainty) of correctness from the model
    """
    pipe = pipeline("text-classification", model="../train_test/finetuned_model", tokenizer="digitalepidemiologylab/covid-twitter-bert-v2")
    result = pipe(input)
    resultdict = result[0]
    label = resultdict['label']
    score = resultdict['score']
    print("** Results **")
    print("Determination: "+label)
    print("Certainty: "+str(score))      
    return label, score