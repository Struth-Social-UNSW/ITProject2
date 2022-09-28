""" dl_preproc_dummy_enc.py: 
    
    This program dummy encodes the labels for the deep learning model, should the data
    not be supplied in this format. 
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "28 Sep 22"
__version__     = 1.0   
__status__      = "Complete"
__notes__       = "This program is intended to be run as a standalone program"


# importing required libraries
import pandas as pd
    
def rewritelabels(file):
    """ This function prepares the input for the deep learning model by encoding the labels
        in a binary manner
            -real is 0
            -fake is 1
        
        On completion of the encoding, the file is output and now ready for further preprocessing

        ** Parameters **
        file: a str being the directory location of the file to be encoded
    """
    file = pd.read_csv(file) # reading the file for encoding
    file['label'] = file['label'].replace(['TRUE','FAKE'],[0,1])    # encoding the labels
    file.to_csv('preproc_data_enc.csv', index=False)    # saving both train and test files after encoding

rewritelabels("preproc_data.csv")