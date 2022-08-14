#
#
# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
nltk.download('stopwords')
import itertools

# Import libraries from scikit learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Import libraries from ntlk
from nltk.corpus import stopwords
from nltk import tokenize

# Import libraried from WordCloud
from wordcloud import WordCloud

# Load and read data from csv file
dataset = pd.read_csv('kaggle-covid-news.csv')
print('Dataset shape: ', dataset.shape)

### Pre-processing ###

# Determine weight of dataset
countFalse = dataset['label'].value_counts()
print(countFalse)

countFalse = dataset['label'].value_counts('false')
print(countFalse)

# Add flag to track fake and real articles
dataset['target1'] = 'fake'
dataset['target2'] = 'true'

# Remove any unknown or unlabeled rows
dataset.drop(dataset[dataset['label'] == 'U'].index, inplace = True)
#print('\nDataset.head: \n', dataset.head())

# Convert to lowercase
dataset['text'] = dataset['text'].apply(lambda x: x.lower())
#print('\nDataset.head: \n', dataset.head())

# Remove punctuation
def punctuation_removal(text):
    all_list = [char for char in  text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

dataset['text'] = dataset['text'].apply(punctuation_removal)
#print('\nDataset.head: \n', dataset.head())

# Remove stopwords
stop = stopwords.words('english')
dataset['text'] = dataset['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print('\nDataset.head: \n', dataset.head())

### Data Exploration ###

# Word cloud for fake news
fake_data = dataset[dataset["target1"] == "fake"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width = 800,
                        height = 500,
                        max_font_size = 110,
                        collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
#plt.show()

# Word cloud for real news
real_data = dataset[dataset["target2"] == "real"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width = 800,
                        height = 500,
                        max_font_size = 110,
                        collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
#plt.show()

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
    ax = sns.barplot(data = df_frequency, x="Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation = 'vertical')
    #plt.show()

# Most frequent words in fake news
counter(dataset[dataset['target1'] == "fake"], "text", 20)

# Most frequent words in real news
counter(dataset[dataset['target2'] == "true"], "text", 20)

### Modeling ###

# Confustion matrix
def plot_confusion_matrix(cm, classes,
                            normalize = False,
                            title = 'Confusion matrix', 
                            cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalised confusion matrix')
    else:
        print('Confusion matrix, without normalisation')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

### Preparing data ###

# Divide data for training and testing (currently 80:20 - train:test)
x_train, x_test, y_train, y_test = train_test_split(dataset['text'], dataset['label'], test_size = 0.2, random_state = 42)


### Logistic regression ###

# Vectorising and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()), 
                    ('model', LogisticRegression())])

# Fitting the model
model = pipe.fit(x_train, y_train)

# Accuracy
pred = model.predict(x_test)

# Calculate accuracy of model over testing data
score = accuracy_score(y_test, pred)
print('\n*** Logistic regression ***')
print("Accuracy: ", round(score*100,2), "%")

# Confusion matrix
print(confusion_matrix(y_test, pred))
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])


### Decision Tree Classifier ###

# Vectorising and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()), 
                    ('model', DecisionTreeClassifier(criterion= 'entropy',
                                                                max_depth=20,
                                                                splitter='best',
                                                                random_state=42))])

# Fitting the model
model = pipe.fit(x_train, y_train)

# Accuracy
pred = model.predict(x_test)

# Calculate accuracy of model over testing data
score = accuracy_score(y_test, pred)
print('\n*** Decision Tree Classifier ***')
print("Accuracy: ", round(score*100,2), "%")

# Confusion matrix
print(confusion_matrix(y_test, pred))
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])


### Random Forest Classifier ###

# Vectorising and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()), 
                    ('model', RandomForestClassifier(n_estimators=50,
                                                                criterion='entropy'))])

# Fitting the model
model = pipe.fit(x_train, y_train)

# Accuracy
pred = model.predict(x_test)

# Calculate accuracy of model over testing data
score = accuracy_score(y_test, pred)
print('\n*** Random Forest Classifier ***')
print("Accuracy: ", round(score*100,2), "%")

# Confusion matrix
print(confusion_matrix(y_test, pred))
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])