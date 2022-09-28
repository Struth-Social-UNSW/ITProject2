""" dl_model_train.py: 
    
    This program provides the interface for the model to be trained. It contains the functions necessary to train the model, as well as the main function which calls the other functions.
    The functions are as follows:
    - tokenize_function: This function tokenizes the text and labels for the model
    - prep: This function prepares the preprocessed data for learning by the model. This involves:
        - Loading the data
        - Creating a dataset object for the model to use
        - Splitting the data into training and testing sets
        - Establishing a separate validation set for confirmation of training statistics
        - Tokenizing all data
    - compute_metrics: This function computes the metrics for the model. This includes:
        - Accuracy
        - Recall
        - Precision
        - F1 Score
    - dlmodelmain: This function contains the model which takes in a string or list of strings and performs an analysis of that text
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "28 Sep 22"
__Version__     = 3.0
__status__      = "Complete"
__notes__       = "This program is intended to be driven from dl_backend_train.py"

# importing required libraries
# !pip install transformers
# !pip install datasets
# !pip install tensorflow
# !pip install numpy
# !pip install pandas
# !pip install sklearn.metrics
# !pip install torch
# !pip install huggingface_hub
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

notebook_login()    # logs into the Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=128)    # loads the tokenizer
metric = load_metric('accuracy')    
model = ""  # creates a global variable for the model
training_args = ""      # creates a global variable for the training arguments
trainer = ""    # creates a global variable for the trainer
token_train = []    # creates a global variable for the training set
token_test = []     # creates a global variable for the testing set     
predtokens = []     # creates a global variable for the validation set
    
def tokenize_function(examples):
    """ This function tokenizes the text and labels for the model

        ** Parameters **
        examples: a row from the dataset object containing the text and label
        
        ** Returns **
        tokenizer: A token object containing the text and label, as well as the tokenized text
    """
    return tokenizer(examples["text"], truncation=True)

def prep():
    """ This function prepares the preprocessed data for learning by the model. This involves:
        - Loading the data
        - Creating a dataset object for the model to use
        - Splitting the data into training and testing sets
        - Establishing a separate validation set for confirmation of training statistics
        - Tokenizing all data

        ** Parameters **
        N/A
        
        ** Returns **
        N/A
    """
    dataset = load_dataset('csv', data_files="preproc_data_done.csv", split="train").shuffle(seed=42)   # loads the train/test set
    dataset_split = dataset.train_test_split(test_size=0.2)   

    preddataset = load_dataset('csv', data_files="preproc_data_test.csv", split="train")    # loads the separate validation set

    global predtokens
    predtokens = preddataset.map(tokenize_function, batched=True)   # tokenizes the validation set
    
    tokens = dataset_split.map(tokenize_function, batched=True) # tokenizes the train/test set
    
    global token_train
    token_train = tokens['train']  # creates a dataset object for the training set
    global token_test
    token_test = tokens['test']   # creates a dataset object for the testing set

def compute_metrics(eval_pred):
    """ This function computes the metrics for the model. This includes:
        - Accuracy
        - Recall
        - Precision
        - F1 Score

        ** Parameters **
        eval_pred: The predictions of the model for evaluation
        
        ** Returns **
        metrics: A dictionary containing the metrics for the model
    """
    pred, labels = eval_pred    # unpacks the predictions and labels
    pred = np.argmax(pred, axis=1)  # gets the index of the highest probability

    accuracy = accuracy_score(y_true=labels, y_pred=pred)   # computes the accuracy
    recall = recall_score(y_true=labels, y_pred=pred)   # computes the recall
    precision = precision_score(y_true=labels, y_pred=pred)  # computes the precision
    f1 = f1_score(y_true=labels, y_pred=pred)   # computes the f1 score

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}   # returns the metrics

def finetune():
    """ This function performs the finetuning of the model. This includes:
    - Creating the model
    - Creating the training arguments
    - Creating the Trainer Class
    - Training the model
    - Evaluating the model
    - Saving the model
    - Uploading the model to HuggingFace Hub

        ** Parameters **
        N/A
        
        ** Returns **
        N/A
    """
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)   # creates the model
    training_args = TrainingArguments(  # creates the training arguments
        "bvrau/covid-twitter-bert-v2-struth",
        evaluation_strategy="epoch", # Frequency of evaluating
        logging_strategy="epoch", # Frequency of logging to file
        save_strategy="epoch", # Frequency of saving checkpoints
        overwrite_output_dir=True,  # Overwrites the output directory
        learning_rate=2e-5, # Learning rate
        per_device_train_batch_size=16, # Batch size for training
        per_device_eval_batch_size=16,  # Batch size for evaluation
        num_train_epochs=50, # could consider doing 10+ epochs
        weight_decay=0.01,  # Weight decay
        load_best_model_at_end=True,    # Loads the best model at the end of training
        push_to_hub=True,   # Pushes the model to HuggingFace Hub
    )
    
    trainer = Trainer(  # creates the Trainer Class
        model=model,    # Passes the model
        args=training_args, # Passes the training arguments
        train_dataset=token_train,  # Passes the training set
        eval_dataset=token_test,    # Passes the testing set
        tokenizer=tokenizer,    # Passes the tokenizer
        compute_metrics=compute_metrics,    # Passes the metrics
    ) 
    
    train = trainer.train() # trains the model
    print(train)    # prints the training statistics
    eval = trainer.evaluate()   # evaluates the model
    print(eval) # prints the evaluation statistics
    preds = trainer.predict(test_dataset=predtokens)    # predicts the validation set
    print("Preds: ", preds) # prints the predictions
    trainer.save_model("finetuned_model")   # saves the model
    trainer.push_to_hub("End of training")  # pushes the model to HuggingFace Hub
    

def dlmodelmain():
    """ This function contains the model which takes in a string or list of strings and performs an analysis of that text

        ** Parameters **
        N/A

        ** Returns **
        N/A
    """
    # print('Rewriting label values')
    # rewritelabels()
    print("Beginning prep tasks")
    prep()
    print("Beginning finetune tasks")
    finetune()

dlmodelmain()