""" dl_model.py: 
    
    This program provides the access to the model and generates a many dimensions vector which models
    the relationship of the text between itself as well as the meaning of the text hollistically
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "12 Aug 22"
__Version__     = 1.0

# importing required libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import tensorflow as tf
import numpy as np
import pandas as pd     # for the formatting/reading of data

tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2") 
metric = load_metric('accuracy')
model = ""
training_args = ""
trainer = ""
token_train = []
token_test = []

    
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def prep():
    dataset = load_dataset('csv', data_files={'train': 'preproc_data_train.csv', 'test': 'preproc_data_test.csv'})
    print(len(dataset))
    print(dataset['train'][1])
    
    tokens = dataset.map(tokenize_function, batched=True)
    
    token_train = tokens['train']
    token_test = tokens['test']

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def dlmodel(input):
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        input: a str containing the body of a Tweet (after being preprocessed)

        ** Returns **
        A many dimensional vector modelling the features of the text, as per the pretraining of the model
    """
    pipe = pipeline("zero-shot-classification", model="digitalepidemiologylab/covid-twitter-bert-v2")    
    fake_real = ['fake', 'real']
    statement = 'This example is {}.'
    result = pipe(input, fake_real, hypothesis_template=statement, multi_label=True)
    return result

def finetune():
    model = AutoModelForSequenceClassification.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2", num_labels=2)
    training_args = TrainingArguments(output_dir="trg_data", evaluation_strategy="epoch")
    
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=token_train,
    eval_dataset=token_test,
    compute_metrics=compute_metrics,
    ) 
    
    trainer.train()
    trainer.evaluate()
    trainer.save_model("finetuned_model")
    
    
    
    
    
    


        
   

def dlmodelmain():
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        N/A

        ** Returns **
        N/A
    """
    dlmodeltrain()
    
    
    
    
    
    