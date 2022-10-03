""" dl_model_eval_logits.py: 
    
    This program is used to separately evaluate the model with the eval dataset. It returns the raw logits
    from the model which are then used to calculate the certainty of the model's prediction.
    The methods used to generate this vector are based on the BERT model and the transformers library.
    - BERT: Bidirectional Encoder Representations from Transformers
    - transformers: https://huggingface.co/transformers/
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "28 Sep 22"
__Version__     = 1.0
__status__      = "Complete"
__notes__       = "This program is intended to be run as a standalone program"

# importing required libraries
# !pip install transformers
# !pip install scipy
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import pandas as pd
from scipy.special import softmax
import torch
import csv

model = AutoModelForSequenceClassification.from_pretrained("bvrau/covid-twitter-bert-v2-struth", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bvrau/covid-twitter-bert-v2-struth", model_max_length=128)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

def dlmodelmain(inputstr):
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        input: a str containing the body of a Tweet (after being preprocessed)

        ** Returns **
        N/A
    """
    predframe = pd.DataFrame({"text": [inputstr]})  # creates a pandas DataFrame containing the input string

    predset = Dataset.from_pandas(predframe)  # creates a Transformers Dataset object from the DataFrame

    predtoken = predset.map(tokenize_function, batched=True)  # tokenizes the DataFrame

    trainer = Trainer(model=model)  # creates a Trainer object from the model
    test_results = trainer.predict(predtoken)  # predicts the label for the input string
    print(test_results) # prints the results of the prediction

    logits = test_results[0][0] # extracts the logits from the prediction

    cert = softmax(logits, axis=0)  # calculates the certainty percentages of the prediction using a softmax function
    print(cert)  

    if(cert[0] > cert[1]):  # determines the label of the prediction
      print("This news is real")
      print("Certainty ", float(cert[0]))
    elif(cert[0] < cert[1]):  
      print("This news is fake")
      print("Certainty: ", float(cert[1]))
      
    # alternate prediction pipeline - not in use
    # pipe = pipeline("text-classification", model=model, tokenizer="/content/results/checkpoint-2532", top_k=2, function_to_apply="sigmoid")
    # result = pipe(input)
    # print(result)
    # resultdict = result[0]
    # label = resultdict['label']
    # score = resultdict['score']
    # print("** Results **")
    # print("Determination: "+label)
    # print("Certainty: "+str(score))      
    # return label, score

def dataloader(file):
  """ This function loads the data from the csv file, lists expected values and provides
      strings to the dlmodelmain function for testing

        ** Parameters **
        input: a str containing the body of a Tweet (after being preprocessed)

        ** Returns **
        N/A
    """
  test = file
  with open(test, 'r') as csvfile:  # opens the csv file
    datareader = csv.reader(csvfile, delimiter=',') # creates a csv reader object
    for row in datareader:  # iterates through the rows of the csv file
      print(row)
      if row[0] == 0: # determines the expected label of the prediction
        actual = "real"
      else:
        actual = "fake"
      print("This should be classified: ", actual)  # prints the expected label
      dlmodelmain(row[1]) # calls the dlmodelmain function to perform the prediction

dataloader("preproc_data_test.csv")