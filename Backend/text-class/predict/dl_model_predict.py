""" dl_model_predict.py: 
    
    This program provides the access to the model through a prediction pipeline.
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "28 Sep 22"
__version__     = 2.0
__status__      = "Complete"
__notes__       = "This program is intended to be driven from the dl_backend_predict.py program"

# importing required libraries
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def dlmodelmain(input):
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        input: a str containing the body of a Tweet (after being preprocessed)

        ** Returns **
        label: a str (either fake or real) based on the model's classification
        score: the score (certainty) of correctness from the model
    """
    model = AutoModelForSequenceClassification.from_pretrained("bvrau/covid-twitter-bert-v2-struth")    # this is the model that was trained
    tokenizer = AutoTokenizer.from_pretrained("bvrau/covid-twitter-bert-v2-struth", truncation=True, padding="max_length", return_tensors="pt")  # this is the tokenizer that was used to train the model
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)    # this is the pipeline that is used to make predictions
    result = pipe(input)    # this is the prediction
    resultdict = result[0]  # this is the dictionary containing the label and score
    label = resultdict['label'] # this is the label
    score = resultdict['score'] # this is the score
    print("** Results **")  # prints the results
    print("Determination: "+label)
    print("Certainty: "+str(score))      
    return label, score  # returns the label and score