import spacy    # text preprocessing utility
import re       # regex ops utility
import html     # for the resolution of HTML entities

orig_text = "As of 18 August 2020 8AM till now there have been a total of 4687 #COVID19 positive cases &amp; 17 #COVID_19 related deaths in #Manipur #COVID__19 #COVIDー19 #COVID19India #COVIDUpdates #CoronaUpdates #Corona #coronavirus #CoronavirusIndia #CoronavirusUpdates #CoronavirusPandemic https://t.co/au9kzAchGh �"

## Removing URLs/Emojis
remurl = re.sub('http://\S+|https://\S+', '', orig_text)

## Removing Twitter Handles
remhand = re.sub('@[^\s]+', '', remurl)

## Removing Hashtags
remhash = re.sub('#[^\s]+', '', remhand)

## Removing non-ASCII characters
nonascii = remhash.encode("ascii", "ignore")
remnonascii = nonascii.decode()

# Removing HTML entities
remhtml = html.unescape(remnonascii)

# Cleaning up double spaces created in removal
remspc = re.sub(' +', ' ', remhtml)

print(remspc)

# loading the basic English library for preprocessing tasks
nlp = spacy.load('en_core_web_sm')

# sets the text being input for preprocessing
text_test = nlp(remhtml)

## Tokenisation
# creates a list to hold the results of the tokenization
token_list = []

# appends the processed tokens to the list
for token in text_test:
    if str(token) != ' ':
        token_list.append(token)

## 

    
print(token_list)