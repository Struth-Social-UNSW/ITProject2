""" dl_model_eval.py: 
    
    This program is used to separately evaluate the model with the eval dataset.
    
    A CSV containing the eval dataset is iterated through and the model is used to predict the label of each Tweet.
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "28 Sep 22"
__version__     = 1.5
__status__      = "Complete"
__notes__       = "This program is intended to be run as a standalone program"

# importing required libraries
# !pip install transformers
from transformers import pipeline, AutoModelForSequenceClassification
import csv

def dlmodelmain(input):
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        input: a str containing the body of a Tweet (after being preprocessed)

        ** Returns **
        N/A
    """
    model = AutoModelForSequenceClassification.from_pretrained("bvrau/covid-twitter-bert-v2-struth", num_labels=2)  # this is the model that was trained
    pipe = pipeline("text-classification", model=model, tokenizer="bvrau/covid-twitter-bert-v2-struth", top_k=2, function_to_apply="sigmoid")   # this is the pipeline that is used to make predictions
    result = pipe(input)    # this is the prediction
    print(result,"\n")      # prints the prediction result
    
    # this section of code is used to return the predicted label and score - not in use
    # resultdict = result[0]
    # label = resultdict['label']
    # score = resultdict['score']
    # print("** Results **")
    # print("Determination: "+label)
    # print("Certainty: "+str(score))      
    # return label, score

def dataloader(file):
    """ This function loads the data from the csv file and lists item by item the expected output.
        The data is then passed to the dlmodelmain function for pipeline prediction.
        The results are then compared to the expected output and the accuracy is calculated.

        ** Parameters **
        N/A

        ** Returns **
        N/A
    """ 
    test = file
    with open(test, 'r') as csvfile:    # opens the csv file
        datareader = csv.reader(csvfile, delimiter=',')   # reads the csv file
        next(datareader)    # skips the first row of the csv file (headers)
        for row in datareader:  # iterates through each row of the csv file
            if row[0] == "0":   
                actual = "real" # sets the expected output to real
            elif row[0] == "1":
                actual = "fake" # sets the expected output to fake
        print("This should be classified: ", actual)    # prints the expected output
        dlmodelmain(row[1]) # passes the Tweet body to the dlmodelmain function for prediction

dataloader("preproc_data_eval.csv")