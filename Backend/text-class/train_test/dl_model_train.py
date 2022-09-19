""" dl_model.py: 
    
    This program provides the access to the model and generates a many dimensions vector which models
    the relationship of the text between itself as well as the meaning of the text hollistically
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "12 Aug 22"
__Version__     = 1.0

# importing required libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, load_metric
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2", model_max_length=128) 
metric = load_metric('accuracy')
model = ""
training_args = ""
trainer = ""
token_train = []
token_test = []
    
def tokenize_function(examples):
    """ This function tokenizes the preprocessed Tweets for training/validation on the DL model

        ** Parameters **
        examples: a str being the preprocesed Tweet
        
        ** Returns **
        tokenizer object: a list of integers representing token IDs
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def prep():
    """ This reads the procesed Tweets, handles their tokenization and prepares
        them for the training task

        ** Parameters **
        N/A
        
        ** Returns **
        N/A
    """
    dataset = load_dataset('csv', data_files={'train': './preproc_data_train.csv', 'test': './preproc_data_test.csv'})   # loads the preprocessed files 
    print("The size of the preprocessed dataset is: "+str(len(dataset)))
    
    tokens = dataset.map(tokenize_function, batched=True)   # tokenizes the preprocessed examples
    
    global token_train
    token_train = tokens['train']   # on tokenization completion, creates separate lists for train and test values
    global token_test
    token_test = tokens['test']

def compute_metrics(eval_pred):
    """ This method provides metrics for the training and testing tasks

        ** Parameters **
        eval_pred: dataset from that instance of training (usually an epoch worth)
        
        ** Returns **
        metric object: A dict object containing calculated loss (training and evaluation) as well as other metrics and the epoch number
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def finetune():
    """ This method conducts the finetuning task on the model.
    
        Begins by defining training arguments (hyperparameters) for the training task. The trainer is then defined as a class, and
        key variables are defined within the class declaration.
        
        Finally, the training task is run, an evaluation is conducted and then the newly produced model is saved.

        ** Parameters **
        N/A
        
        ** Returns **
        N/A
    """
    model = AutoModelForSequenceClassification.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2", num_labels=2)    # importing the model for finetuning
    # declaration of args (hyperparameters) for finetuning the model
    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5, # could consider doing 10+ epochs
        weight_decay=0.01,
    )
    
    # declaration of the Trainer class for finetuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=token_train,
        eval_dataset=token_test,
        compute_metrics=compute_metrics,
    ) 
    
    # training, evaluation and saving of the finetuned model
    trainer.train()
    trainer.evaluate()
    trainer.save_model("finetuned_model")

def dlmodelmain():
    """ This function drives the dl_model_train.py script to conduct the required tasks
    
        Firstly, the data is prepared for finetuning. After preparation is complete, control is handed off to the
        finetune method for further work.
        
        When the finetuning is complete, the newly created model is output to the specified directory.

        ** Parameters **
        N/A

        ** Returns **
        N/A
    """
    prep()
    finetune()
    
    
    
    
    
    