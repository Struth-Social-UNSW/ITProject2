# This program reads data, conducts ML training and testing, outputing an accuracy level and confusion matrix.
# 
# NOTE: The dataset used must contain real/fake labelling
#
#
# Import modules
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
nltk.download('stopwords')
import itertools
import pickle

# Import libraries from scikit learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Import libraries from ntlk
from nltk.corpus import stopwords
from nltk import tokenize

# Import libraried from WordCloud
from wordcloud import WordCloud


###########################################################################

#####  Load & Read Dataset  #####

###  Read dataset  ###
def read(dataFile):
    # Load and read data from csv file
    dataset = pd.read_csv(dataFile)
    print('Dataset shape: ', dataset.shape)

    return dataset


###########################################################################

#####  Pre-Processing  #####


###  Remove punctuation  ###
def punctuation_removal(text):
    all_list = [char for char in  text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


###  Pre-process dataset  ###
def preprocess(dataset):

    dataset['label'].mask(dataset['label'] == 1, 'real', inplace=True)
    dataset['label'].mask(dataset['label'] == 0, 'fake', inplace=True)
    dataset['label'].mask(dataset['label'] == 'T', 'real', inplace=True)
    dataset['label'].mask(dataset['label'] == 'F', 'fake', inplace=True)
    dataset['label'].mask(dataset['label'] == 'REAL', 'real', inplace=True)
    dataset['label'].mask(dataset['label'] == 'TRUE', 'real', inplace=True)
    dataset['label'].mask(dataset['label'] == 'FAKE', 'fake', inplace=True)
    dataset['label'].mask(dataset['label'] == 'Real', 'real', inplace=True)
    dataset['label'].mask(dataset['label'] == 'True', 'real', inplace=True)
    dataset['label'].mask(dataset['label'] == 'Fake', 'fake', inplace=True)

    # Determine weight of dataset
    countFalse = dataset['label'].value_counts()
    print(countFalse)

    countFalse = dataset['label'].value_counts('fake')
    print(countFalse)

    # Add flag to track fake and real articles
    #dataset['target1'] = 'fake'
    #dataset['target2'] = 'true'

    # Remove any unknown or unlabeled rows
    dataset.drop(dataset[dataset['label'] == 'U'].index, inplace = True)

    # Remove any rows with null values 
    dataset.dropna(inplace = True)

    print('\nDataset.head: \n', dataset.head())

    # Convert to lowercase
    #dataset['title'] = dataset['title'].apply(lambda x: x.lower())
    dataset['text'] = dataset['text'].apply(lambda x: x.lower())
    #print('\nDataset.head: \n', dataset.head())

    # Remove punctuation
    #dataset['title'] = dataset['title'].apply(punctuation_removal)
    dataset['text'] = dataset['text'].apply(punctuation_removal)
    #print('\nDataset.head: \n', dataset.head())

    # Remove URL starting with 'http' or 'https', special characters and tags
    #dataset['text'] = dataset['text'].str.replace(r's*https?://S+(s+|$)', ' ').str.strip()
    dataset['text'] = dataset['text'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')
    dataset['text'] = dataset['text'].str.replace(r"[\"\'\|\?\=\.\<\>\@\#\*\{\}\_\,]", '')

    # Remove emojis
    filter_char = lambda c: ord(c) < 256
    dataset['text'] = dataset['text'].apply(lambda s: ''.join(filter(filter_char, s)))
    #print('\nDataset.head: \n', dataset.head())

    # Remove stopwords
    stop = stopwords.words('english')
    #dataset['title'] = dataset['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    dataset['text'] = dataset['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    print('\nDataset.head: \n', dataset.head())

    # Shuffle dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    #df4 = df4.sample(frac=1)

    # Print first 5 rows of dataset after pre-processing
    print('\nDataset.head shuffled: \n', dataset.head())

    return dataset


###########################################################################

#####  Data Exploration  #####


###  Word cloud for FAKE news  ###
def fakeCloud(dataset):
    fake_data = dataset[dataset["label"] == "fake"]
    all_words = ' '.join([text for text in fake_data.text])

    wordcloud = WordCloud(width = 800,
                            height = 500,
                            max_font_size = 110,
                            collocations = False).generate(all_words)

    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # Saves word cloud as jpg
    plt.savefig('img_fakeCloud.jpg')

    # Displays word cloud to screen
    #plt.show()
    plt.close()


###  Word cloud for REAL news  ###
def realCloud(dataset):
    real_data = dataset[dataset["label"] == "real"]
    all_words = ' '.join([text for text in real_data.text])

    wordcloud = WordCloud(width = 800,
                            height = 500,
                            max_font_size = 110,
                            collocations = False).generate(all_words)

    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # Saves word cloud as jpg
    plt.savefig('img_realCloud.jpg')

    # Displays word cloud to screen
    #plt.show()
    plt.close()


###  Most frequent words counter  ###
def counter(text, column_text, quantity, token_space):
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

    # Saves frequent words table as jpg
    plt.savefig('img_freqWords.jpg')

    # Displays frequent words table to screen
    plt.show()
    plt.close()


###  Count most frequent words in fake and real news  ###
def countWords(dataset):
    token_space = tokenize.WhitespaceTokenizer()

    # Most frequent words in fake news
    counter(dataset[dataset['label'] == "fake"], "text", 20, token_space)

    # Most frequent words in real news
    #counter(dataset[dataset['target2'] == "true"], "text", 20, token_space)


###########################################################################

#####  Modelling  #####


###  Calculate accuracy of model over testing data  ###
def accuracy(y_test, predicted):
    score = accuracy_score(y_test, predicted)
    print("Accuracy: ", round(score*100,2), "%")
    #f1Score = f1_score(y_test, predicted)
    #print("F1 Score: ", round(f1Score*100,2), "%")
    print("Precision:   measures the proportion of positively predicted labels that are acutualy correct.")
    print("Recall:      represents the model's ability to correctly predict the positives out of actual positives.")
    print("F1 Score:    represents the model score as a function of precision and recall score.")
    print("Support:     number of actual occurences of the label in the dataset.\n")
    print(metrics.classification_report(y_test, predicted))


###  Confustion matrix  ###
def plotConfusionMatrix(cm, classes,
                            normalize = False,
                            title = 'Confusion matrix', 
                            cmap = plt.cm.Blues, nom = ''):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
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
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Saves confusion matrix as jpg
    plt.savefig('img_confMatrix_' + nom + '.jpg')

    # Displays confusion matrix to the screen
    plt.show()
    plt.close()


###   Display Confustion Matrix  ###
def dispConfusionMatrix(y_test, predicted, classifer, model):
    print(metrics.confusion_matrix(y_test, predicted))
    cm = metrics.confusion_matrix(y_test, predicted)
    plotConfusionMatrix(cm, classes=['Fake', 'True'], title = classifer, nom = model)  ## Uncomment this line to save/display confusion matrix


###########################################################################

#####  Data Preparation  #####


###  Prepare data  ###
def prepareData(dataset):
    # Divide data for training and testing (currently 80:20 - train:test)
    x_train, x_test, y_train, y_test = train_test_split(dataset['text'], dataset['label'], test_size = 0.2, random_state = 42)

    return x_train, x_test, y_train, y_test
    

###########################################################################

#####  Classifiers  #####


###  Pipeline and GridSearch Classifier  ###
def pipeGrid(dataset):
    
    classifierBest = 'TF-IDF Best Model Confusion Matrix'
 
    print('\n***  Pipeline and GridSearch  ***')
   
    # Create pipeline
    pipe = Pipeline(steps = [('tfidf_vectorization', TfidfVectorizer()), ('classifier', MultinomialNB)])

    # Create dictionary with hyperparameters
    search_space = [{'classifier': [MultinomialNB()]},
                    {'classifier': [LinearSVC()]},
                    {'classifier': [PassiveAggressiveClassifier()]},
                    {'classifier': [LogisticRegression()]},
                    {'classifier': [DecisionTreeClassifier()]},
                    {'classifier': [RandomForestClassifier()]},
                    {'classifier': [LogisticRegression()],'classifier__solver': ['liblinear']},
                    {'classifier': [KNeighborsClassifier()], 'classifier__n_neighbors': [5,6,7,8]}]
    
    # Create the GridSearchCV object, Area Under the Curve of the Receiver Operating Characteristics curve
    scoring = {'AUC': 'roc_auc', 'Accuracy': metrics.make_scorer(metrics.accuracy_score)}
    grid = GridSearchCV(estimator = pipe, param_grid = search_space, cv = 10, scoring = scoring, return_train_score = True, n_jobs = -1, refit = 'AUC')

    # Fit GridSearch object
    best_model = grid.fit(x_train, y_train)
    print('Best: %f using %s' % (best_model.best_score_, best_model.best_params_))

    best_model_pred = best_model.predict(x_test)

    # Calculate accuracy of model over testing data & confusion matrix
    print('\n*** TF-IDF Best Model ***')

    accuracy(y_test, best_model_pred)
    dispConfusionMatrix(y_test, best_model_pred, classifierBest, 'BEST-TF-IDF')
    

    return best_model


    

    



###########################################################################

#####  Main Program  #####


# Dataset source
dataFile = './kaggle-covid-news.csv'
#dataFile = './general-news.csv'
#dataFile = './covid-news.csv'
#dataFile = './general-WELFake.csv'
#dataFile = './preproc_combo.csv'

# Load and read dataset
data = read(dataFile)

# Conduct preprocessing of dataset
data = preprocess(data)

# Create images/plots of data
fakeCloud(data)
realCloud(data)
countWords(data)

# Prepare data for training and testing
x_train, x_test, y_train, y_test = prepareData(data)

# Execute Classifiers
pipeGridModel = pipeGrid(data)


##########################################################################

##  Query Tweet/headline/text for fake news determination on Covid

# News article input
news = 'covid is hoax'
print('\nNews article reads: ', news, '\n')

# Pipeline GridSearch result
result = pipeGridModel.predict([news])
#conf = naiveBayesModel.predict_proba([news])
#print('Naive Bayes result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
print('Pipeline GridSearch result is: \t', result[0])

########################################

# News article input
news = 'washing your hands regularly is one of the best ways to prevent the spead of coronavirus'
print('\nNews article reads: ', news, '\n')

# Pipeline GridSearch result
result = pipeGridModel.predict([news])
print('Pipeline GridSearch result is: \t', result[0])

########################################

# News article input
news = 'The novel coronavirus outbreak has spread to more than 150 countries or territories around the world'
print('\nNews article reads: ', news, '\n')

# Pipeline GridSearch result
result = pipeGridModel.predict([news])
print('Pipeline GridSearch result is: \t', result[0])

########################################

# News article input
news = 'Coronavirus is only dangerous for old people'
print('\nNews article reads: ', news, '\n')

# Pipeline GridSearch result
result = pipeGridModel.predict([news])
print('Pipeline GridSearch result is: \t', result[0])

##########################################################################


##  Query Tweet/headline/text for fake news determination on general news

# News article input
news = 'NASA is installing internet on the moon'
print('\nNews article reads: ', news, '\n')

# Pipeline GridSearch result
result = pipeGridModel.predict([news])
print('Pipeline GridSearch result is: \t', result[0])


########################################

# News article input
news = 'Spinach is taught how to send emails'
print('\nNews article reads: ', news, '\n')

# Pipeline GridSearch result
result = pipeGridModel.predict([news])
print('Pipeline GridSearch result is: \t', result[0])


########################################

# News article input
news = 'Donald Trump is President'
print('\nNews article reads: ', news, '\n')

# Pipeline GridSearch result
result = pipeGridModel.predict([news])
print('Pipeline GridSearch result is: \t', result[0])


########################################

# News article input
news = 'Donald Trump is a liar'
print('\nNews article reads: ', news, '\n')

# Pipeline GridSearch result
result = pipeGridModel.predict([news])
print('Pipeline GridSearch result is: \t', result[0])


