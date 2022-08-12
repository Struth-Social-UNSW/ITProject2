"""preproc.py: This program provides the preprocessing facility for the Fake News Detection DL model and subsequent ML models."""

__author__      = "Breydon Verryt-Reid"
__date__        = "12 Aug 22"
__Version__     = 1.0

# file imports for required libraries
import spacy    # text preprocessing utility
import re       # regex ops utility
import html     # for the resolution of HTML entities
import emoji    # for conversion of emojis

orig_text = "As of 18 August 2020 8AM till now there have been a total of 4687 #COVID19 positive cases &amp; 17 #COVID_19 related deaths in #Manipur #COVID__19 #COVIDー19 #COVID19India #COVIDUpdates #CoronaUpdates #Corona #coronavirus #CoronavirusIndia #CoronavirusUpdates #CoronavirusPandemic https://t.co/au9kzAchGh � 😂"

def twitter_cleaning(input):
    """ This function prepares the input for preprocessing by removing Twitter specific
        data such as:
            -URLs
            -Twitter handles
            -Hashtags
        
        This function also converts emojis into text, for example:
            -😂 becomes :face_with_tears_of_joy:
    """
    ## Removing URLs
    remurl = re.sub('http://\S+|https://\S+', '', orig_text)

    ## Removing Twitter Handles
    remhand = re.sub('@[^\s]+', '', remurl)

    ## Removing Hashtags
    remhash = re.sub('#[^\s]+', '', remhand)

    ## Switching Emojis to their descriptions
    rememoji = emoji.demojize(remhash)

    return rememoji

def general_cleanup(input_text):
    """ This function prepares the input for preprocessing by tidying general
        data such as:
            -removing non-ASCII characters
            -removing HTML entities
            -removing additional spaces created during preprocessing preparation
    """
    ## Removing non-ASCII characters
    nonascii = rememoji.encode("ascii", "ignore")
    remnonascii = nonascii.decode()

    # Removing HTML entities
    remhtml = html.unescape(remnonascii)

    # Cleaning up double spaces created in removal
    remspc = re.sub(' +', ' ', remhtml)

    print(remspc)

    return remspc

def spacy_preproc(input_text):
    """ This function enables the preprocessing of the Tweet using the spaCy libraries.
        First, the spaCy language model is loaded into the program, before tokenization occurs.
        This exchanges the string values into tokens which can be used again by spaCy.

        Excess spaces are then removed and colons removed. The colons appear from the conversion
        of emojis to text representation. Finally, the tokens are lemmatised before they are
        output as str objects to a new list.
    """
    # loading the basic English library for preprocessing tasks
    nlp = spacy.load('en_core_web_sm')
    stopword_list = nlp.Defaults.stop_words

    # sets the text being input for preprocessing
    text_test = nlp(remspc)

    ## Tokenisation
    # creates a list to hold the results of the tokenization
    token_list = []

    # appends the processed tokens to the list, removing any spaces or colons
    for token in text_test:
        if str(token) != ' ':
            token_list.append(token)

    print('Pre colon removal\br')
    print(token_list)

    ## removes any colons which enter the token list
    for tokens in token_list:
        if str(tokens) == ':':
            token_list.remove(tokens)

    print('Post colon removal\br')
    print(token_list)

    # list for the removed stopwords
    stopwords_rem = []

    # removes stopwords from the text
    for token in token_list:
        if str(token) not in stopword_list:
            stopwords_rem.append(token)

    for token in stopwords_rem:
        if str(token) == '&':
            stopwords_rem.remove(token)

    print(stopwords_rem)

    # creating a list of lemmatised tokens from the sentence
    lemma_text = []

    # adds lemmatised version of all words (if applicable) into a final list, ready for the model
    for word in stopwords_rem:
        lemma_text.append(word.lemma_)

    print(lemma_text)

def main(input_text):
    """The main function for this program. Controls the I/O and flow of program execution"""
    twitter_cleaned = twitter_cleaning(input_text)
    general_cleaned = general_cleanup(twitter_cleaned)
    spacy_cleaned = spacy_preproc(general_cleaned)
    return spacy_cleaned