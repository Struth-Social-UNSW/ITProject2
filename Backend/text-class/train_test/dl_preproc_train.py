"""dl_preproc_train.py: This program provides the preprocessing facility for the Fake News Detection DL model.3

    Various functions are provided to prepare the data for preprocessing, including:
        -rewritelabels: This function prepares the input for the deep learning model by dummy encoding the labels
        -twitter_cleaning: This function prepares the input for preprocessing by removing Twitter specific data
        -general_cleanup: This function prepares the input for preprocessing by tidying general data
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "28 Sep 22"
__Version__     = 3.0
__status__      = "Complete"
__notes__       = "This program is to be driven from the dl_backend_train.py program"

# file imports for required libraries
import spacy            # text preprocessing utility
import re               # regex ops utility
import html             # for the resolution of HTML entities
import emoji            # for conversion of emojis
import pandas as pd     # for the formatting/reading of data   

def rewritelabels(train, test):
    """ This function prepares the input for the deep learning model by dummy encoding the labels
            -real is 0
            -fake is 1
        
        On completion of the encoding, the file is output and now ready for further preprocessing

        ** Parameters **
        train: a str being the name of the training file
        test: a str being the name of the testing/validation file
        
        ** Returns **
        N/A
    """
    trg_file = pd.read_csv(train) # reading the trg file for encoding
    print('Printing TRG file before change')        
    trg_file['label'] = trg_file['label'].replace(['real','fake'],[0,1])    # encoding the labels
    test_file = pd.read_csv(test) # reading the testing file for encoding
    print('Printing TEST file before change')       
    test_file['label'] = test_file['label'].replace(['real','fake'],[0,1])  # encoding the labels 
    
    trg_file.to_csv('./text-class/train_test/trg_data/testrun/train_num.csv', index=False)    # saving both train and test files after encoding
    test_file.to_csv("./text-class/train_test/trg_data/testrun/dev_num.csv", index=False)   

def twitter_cleaning(input):
    """ This function prepares the input for preprocessing by removing Twitter specific
        data such as:
            -URLs
            -Twitter handles
            -Hashtags
            -Twitter URL, User and Retweet tags
            -Repeated characters
        
        This function also converts emojis into text, for example:
            -😂 becomes :face_with_tears_of_joy:

        ** Parameters **
        input: a str containing the body of a Tweet
        
        ** Returns **
        remqmarks: A string object containing the original Tweet processed
    """
    ## Removing URLs
    remurl = re.sub('http://\S+|https://\S+', '', input)

    ## Removing Twitter Handles
    remhand = re.sub('@[^\s]+', '', remurl)
    
    ## Removing Hashtags (covid)
    remhash1 = remhand.replace('#covid', 'covid')
    
    ## Removing Hashtags (covid19)
    remhash2 = remhash1.replace('#covid19', 'covid19')
    
    ## Removing Hashtags (coronavirus)
    remhash3 = remhash2.replace('#coronavirus', 'coronavirus')

    ## Removing Hashtags (general)
    remhash4 = re.sub('#[^\s]+', '', remhash3)
    
    ## Removing twitterurl tags
    remtwitterurl = remhash4.replace('twitterurl', '')
    
    ## Removing twitteruser tags
    remtwitteruser = remtwitterurl.replace('twitteruser', '')
    
    ## Removing rt tags
    remrt = remtwitteruser.replace('rt', '')

    ## Switching Emojis to their descriptions
    rememoji = emoji.demojize(remrt)
    
    ## Removing '???' occurences
    remqmarks = rememoji.replace('???', '')
    remqmarks = remqmarks.replace('??', '')
    
    return remqmarks

def general_cleanup(input):
    """ This function prepares the input for preprocessing by tidying general
        data such as:
            -removing non-ASCII characters
            -removing HTML entities
            -removing additional spaces created during preprocessing preparation

        ** Parameters **
        input: a str containing the body of a Tweet

        ** Returns **
        remspc: A string object containing the original Tweet processed 
    """
    ## Removing non-ASCII characters
    nonascii = input.encode("ascii", "ignore")
    remnonascii = nonascii.decode()

    # Removing HTML entities
    remhtml = html.unescape(remnonascii)

    # Cleaning up double spaces created in removal
    remspc = re.sub(' +', ' ', remhtml)

    return remspc

def spacy_preproc(input):
    """ This function enables the preprocessing of the Tweet using the spaCy libraries.
        First, the spaCy language model is loaded into the program, before tokenization occurs.
        This exchanges the string values into tokens which can be used again by spaCy.

        Excess spaces are then removed and colons removed. The colons appear from the conversion
        of emojis to text representation. Finally, the tokens are output as str objects to a new list.

        ** Parameters **
        input: a str containing the body of a Tweet

        ** Returns **
        preproc_str: A string object of the processed original Tweet
    """
    # loading the basic English library for preprocessing tasks
    nlp = spacy.load('en_core_web_sm')
    stopword_list = nlp.Defaults.stop_words

    # sets the text being input for preprocessing
    text_test = nlp(input)

    ## Tokenisation
    # creates a list to hold the results of the tokenization
    token_list = []

    # appends the processed tokens to the list, removing any spaces or colons
    for token in text_test:
        if str(token) != ' ':
            token_list.append(token)

    ## removes any colons which enter the token list
    for tokens in token_list:
        if str(tokens) == ':':
            token_list.remove(tokens)

    # # list for the removed stopwords
    # stopwords_rem = []

    # # removes stopwords from the text
    # for token in token_list:
    #     if str(token) not in stopword_list:
    #         stopwords_rem.append(token)

    # for token in stopwords_rem:
    #     if str(token) == '&':
    #         stopwords_rem.remove(token)

    # # creating a list of lemmatised tokens from the sentence
    # lemma_text = []

    # # adds lemmatised version of all words (if applicable) into a final list, ready for the model
    # for word in token_list:
    #     lemma_text.append(word.lemma_)
    
    # # creating a list of lemmatised tokens from the sentence
    final_text = []
    
    # adds coverts tokens back into str, then adds into a final list, ready for the model
    for word in token_list:
        final_text.append(str(word))

    preproc_str = ' '.join(final_text)

    return preproc_str

def preprocmain(training, testing):
    """The main function for this program. Controls the I/O and flow of program execution"""
    
    # Encoding activity to prepare for DL training
    rewritelabels(training, testing)
    
    # Drops the ID axis from the dataframe, not required for DL tasks
    trg_file = pd.read_csv('./text-class/train_test/trg_data/testrun/train_num.csv')
    trg_file = trg_file.drop('id', axis=1)
    test_file = pd.read_csv('./text-class/train_test/trg_data/testrun/dev_num.csv')
    test_file = test_file.drop('id', axis=1)
    print('Beginning Pre-Processing. The test dataset size is: '+str(len(trg_file.index))+' and the training dataset size is: '+str(len(test_file.index)))
    
    # Establishing the DF for output of preprocessed Tweets
    preproc_dict = {'label': [], 'text': []}
    preproc_data_train = pd.DataFrame(data=preproc_dict)
    preproc_data_test = pd.DataFrame(data=preproc_dict)
    
    # Running preprocessing for the training data
    for index in trg_file.index:
        print('Text '+str(index)+' starting')
        twitter_cleaned = twitter_cleaning(trg_file['text'][index])
        general_cleaned = general_cleanup(twitter_cleaned)
        spacy_cleaned = spacy_preproc(general_cleaned)
        cleaned = {'label': [trg_file['label'][index]], 'text': [spacy_cleaned]}    # creates a new Dict object containing the preprocessed Tweet
        preproc_add = pd.DataFrame(cleaned)
        preproc_data_train_up = pd.concat([preproc_data_train, preproc_add], sort=False)
        preproc_data_train = preproc_data_train_up  # updates the dataframe with the newly preprocessed Tweet
        
    # Running preprocessing for the testing data
    for index in test_file.index:
        print('Text '+str(index)+' starting')
        twitter_cleaned = twitter_cleaning(test_file['text'][index])
        general_cleaned = general_cleanup(twitter_cleaned)
        spacy_cleaned = spacy_preproc(general_cleaned)
        cleaned = {'label': [test_file['label'][index]], 'text': [spacy_cleaned]} # creates a new Dict object containing the preprocessed Tweet
        preproc_add = pd.DataFrame(cleaned)
        preproc_data_test_up = pd.concat([preproc_data_test, preproc_add], sort=False)
        preproc_data_test = preproc_data_test_up    # updates the dataframe with the newly preprocessed Tweet
    
    # Writing the preprocessed data to CSV files
    preproc_data_train.to_csv('./text-class/train_test/trg_data/testrun/preproc_data_train.csv', index=False)
    
    preproc_data_test.to_csv('./text-class/train_test/trg_data/testrun/preproc_data_test.csv', index=False)