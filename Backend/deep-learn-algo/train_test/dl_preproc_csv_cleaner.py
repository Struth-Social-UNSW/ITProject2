""" dl_preproc_csv_cleaner.py: 
    
    This program cleans the preprocessed csv file and removes any rows which contain null values, before exporting it as a cleaned file. 
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "28 Sep 22"
__version__     = 1.0   
__status__      = "Complete"
__notes__       = "This program is intended to be run as a standalone program"
    
# importing required libraries
import pandas as pd

def clean(file):
    """ This function cleans the provided csv file and removes any rows which contain null values, before exporting it as a cleaned file
    
        Note: it is necessary to manually change the file name in the function call to the name of the file you wish to clean

        ** Parameters **
        file: the directory location of the file you wish to clean

        ** Returns **
        N/A
    """
    trg_file = pd.read_csv(file, low_memory=False)   # loads the csv file
    trg_file = trg_file.dropna()    # removes any rows which contain null values
    trg_file.to_csv(file, index=False)   # saves the csv file

clean("preproc_data_trg_test.csv")   # cleans the preprocessed training data