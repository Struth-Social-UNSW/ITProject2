"""preproc.py: This program provides the preprocessing facility for the Fake News Detection DL model and subsequent ML models."""

__author__      = "Breydon Verryt-Reid"
__date__        = "12 Aug 22"
__Version__     = 1.0

# file imports for required libraries
import spacy    # text preprocessing utility
import re       # regex ops utility
import html     # for the resolution of HTML entities
import emoji    # for conversion of emojis
import tensorflow

def twitter_cleaning(input):
    """ This function prepares the input for preprocessing by removing Twitter specific
        data such as:
            -URLs
            -Twitter handles
            -Hashtags
        
        This function also converts emojis into text, for example:
            -😂 becomes :face_with_tears_of_joy:

        ** Parameters **
        input: a str containing the body of a Tweet
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
        A string object containing the original Tweet processed 
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
        A string object of the processed original Tweet
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

def preprocmain(input_text):
    """The main function for this program. Controls the I/O and flow of program execution
    
        ** Parameters **
        input_text: a str containing the body of a Tweet

        ** Returns **
        spacy_cleaned: A string object of the preprocessed original Tweet
    """
    twitter_cleaned = twitter_cleaning(input_text)
    general_cleaned = general_cleanup(twitter_cleaned)
    spacy_cleaned = spacy_preproc(general_cleaned)
    return spacy_cleaned