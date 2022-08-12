import spacy    # text preprocessing utility
import re       # regex ops utility
import html     # for the resolution of HTML entities
import emoji    # for conversion of emojis

orig_text = "As of 18 August 2020 8AM till now there have been a total of 4687 #COVID19 positive cases &amp; 17 #COVID_19 related deaths in #Manipur #COVID__19 #COVIDー19 #COVID19India #COVIDUpdates #CoronaUpdates #Corona #coronavirus #CoronavirusIndia #CoronavirusUpdates #CoronavirusPandemic https://t.co/au9kzAchGh � 😂"

## Removing URLs
remurl = re.sub('http://\S+|https://\S+', '', orig_text)

## Removing Twitter Handles
remhand = re.sub('@[^\s]+', '', remurl)

## Removing Hashtags
remhash = re.sub('#[^\s]+', '', remhand)

## Switching Emojis to their descriptions
rememoji = emoji.demojize(remhash)

## Removing non-ASCII characters
nonascii = rememoji.encode("ascii", "ignore")
remnonascii = nonascii.decode()

# Removing HTML entities
remhtml = html.unescape(remnonascii)

# Cleaning up double spaces created in removal
remspc = re.sub(' +', ' ', remhtml)

print(remspc)

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