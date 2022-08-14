#
#
# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import ntlk
ntlk.download('stopwords')

# Import libraries from scikit learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import libraries from ntlk
from nltk.corpus import stopwords
from nltk import tokenize

# Import libraried from WordCloud
from wordcloud import WordCloud

# Load and read data from csv file
dataset = pd.read_csv('kaggle-covid-news.csv')
print('Dataset shape: ', dataset.shape)

### Pre-processing ###

# Add flag to track fake and real articles
dataset['target1'] = 'fake'
dataset['target2'] = 'true'

# Remove any unknown or unlabeled rows
dataset.drop(dataset[dataset['label'] == 'U'].index, inplace = True)
print('\nDataset.head: \n', dataset.head())

# Convert to lowercase
dataset['text'] = dataset['text'].apply(lambda x: x.lower())
print('\nDataset.head: \n', dataset.head())

# Remove punctuation
def punctuation_removal(text):
    all_list = [char for char in  text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

dataset['text'] = dataset['text'].apply(punctuation_removal)
print('\nDataset.head: \n', dataset.head())

# Remove stopwords
stop = stopwords.words('english')
dataset['text'] = dataset['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print('\nDataset.head: \n', dataset.head())

### Data Exploration ###

# How many fake and real articles?
print('Fake: ', dataset.groupby(['target1'])['label'].count())
print('True: ', dataset.groupby(['target2'])['label'].count())

# Word cloud for fake news
fake_data = dataset[dataset["target"] == "fake"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width = 800,
                        height = 500,
                        max_font_size = 110,
                        collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Word cloud for real news
real_data = dataset[dataset["target"] == "real"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width = 800,
                        height = 500,
                        max_font_size = 110,
                        collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Most frequent words counter
token_space = tokenize.WhitespaceTokenizer()

def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize = (12,8))
    ax = sns.barplot(dataset = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation = 'vertical')
    plt.show()

# Most frequent words in fake news
counter(dataset[dataset['target1'] == "fake"], "text", 20)

# Most frequent words in real news
counter(dataset[dataset['target2'] == "true"], "text", 20)

### Modeling ###

#