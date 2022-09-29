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
    #print('\nDataset.head: \n', dataset.head())

    # Convert to lowercase
    #dataset['title'] = dataset['title'].apply(lambda x: x.lower())
    dataset['text'] = dataset['text'].apply(lambda x: x.lower())
    #print('\nDataset.head: \n', dataset.head())

    # Remove punctuation
    #dataset['title'] = dataset['title'].apply(punctuation_removal)
    dataset['text'] = dataset['text'].apply(punctuation_removal)
    #print('\nDataset.head: \n', dataset.head())

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


###  Passive Aggressive Classifier  ###
def passiveAggressive(x_train, x_test, y_train, y_test):
    classifier = 'Passiver Aggressive Confusion Matrix'
    nom = 'PA'

    # TFIDF-Vectorizor - text array converted to TF-IDF matrix to define importance of keyword
    # TF (Term Frequency) - number of times a word appears in text
    # IDF (Inverse Document Frequency) - measure of how significant a work is in the entire data
    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)

    # Passive Agressive Classifier - is an online learning alogorithm which remains passive for a correct classification and turns aggressive for miscalculations.
    # It updates loss after each iteration and changes weight vector
    #pipe = PassiveAggressiveClassifier(max_iter = 50)
    model = PassiveAggressiveClassifier(max_iter = 50)

    # Fitting the model
    #model = pipe.fit(tfidf_train, y_train)
    model.fit(tfidf_train, y_train)

    # Predictions about testing data
    predicted = model.predict(tfidf_test)

    # Calculate accuracy of model over testing data
    print('\n*** Passive Aggressive Classifier ***')
    accuracy(y_test, predicted)

    # Display confusion matrix
    dispConfusionMatrix(y_test, predicted, classifier, nom)

    # Pipeline utility function to train and transform data to text data
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words = 'english')), 
                            ('model', MultinomialNB())])

    # Fitting the model
    pipeline.fit(x_train, y_train)

    # Pickling is process where object heirarchy is converted into a byte stream
    # Serialize an object hierarchy
    with open('model.pkl', 'wb') as handle:
        pickle.dump(pipeline, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    # De-serialize data stream
    with open('model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    return model


###  Logistic Regression  ###
def logicRegression(x_train, x_test, y_train, y_test):
    classifier = 'Logistic Regression Confusion Matrix'
    nom = 'LR'

    # Vectorising and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()), 
                        ('model', LogisticRegression())])

    # Fitting the model
    model = pipe.fit(x_train, y_train)

    # Accuracy
    predicted = model.predict(x_test)

    # Calculate accuracy of model over testing data
    print('\n*** Logistic regression ***')
    accuracy(y_test, predicted)

    # Display confusion matrix
    dispConfusionMatrix(y_test, predicted, classifier, nom)

    return model

    
###  Decision Tree Classifier  ###
def decisionTree(x_train, x_test, y_train, y_test):
    classifier = 'Decision Tree Confusion Matrix'
    nom = 'DT'

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
    predicted = model.predict(x_test)

    # Calculate accuracy of model over testing data
    print('\n*** Decision Tree Classifier ***')
    accuracy(y_test, predicted)

    # Display confusion matrix
    dispConfusionMatrix(y_test, predicted, classifier, nom)

    return model


###  Random Forest Classifier  ###
def randomForest(x_train, x_test, y_train, y_test):
    classifier = 'Random Forest Confusion Matrix'
    nom = 'RF'

    # Vectorising and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()), 
                        ('model', RandomForestClassifier(n_estimators=50,
                                                                    criterion='entropy'))])

    # Fitting the model
    model = pipe.fit(x_train, y_train)

    # Accuracy
    predicted = model.predict(x_test)

    # Calculate accuracy of model over testing data
    print('\n*** Random Forest Classifier ***')
    accuracy(y_test, predicted)

    # Display confusion matrix
    dispConfusionMatrix(y_test, predicted, classifier, nom)

    return model



###  Multinominal Naive Bayes Classifier  ###
def naiveBayes(dataset):
    classifierTfidf = 'Naive Bayes Tf-idf Confusion Matrix'
    classifierCount = 'Naive Bayes Count Confusion Matrix'
    nom = 'NB'

    # Create target
    y = dataset['label']

    # Divide data for training and testing (currently 80:20 - train:test)
    x_train, x_test, y_train, y_test = train_test_split(dataset['text'], y, test_size = 0.2, random_state = 11)

    # Pre-process data with CountVectorizer and TfidfVectorizor because ML algorithms only work with numerical data
    # CountVectorizer creates dictionary with occurrence number of tokens
    count_vectorizer = CountVectorizer(stop_words='english', min_df = 0.05, max_df = 0.9)
    count_train = count_vectorizer.fit_transform(x_train, y_train)
    count_test = count_vectorizer.transform(x_test)

    # TfidfVectorizer creates dictionary with tf-idf values of tokens
    # It determines the importance of a particular token, if it is common - value will be low, if it is rare - value will be high
    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 0.05, max_df = 0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train, y_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)

    # Create Multinominal Naive Bayes models, train and run predictions
    tfidf_nb = MultinomialNB()
    tfidf_nb.fit(tfidf_train, y_train)
    tfidf_nb_pred = tfidf_nb.predict(tfidf_test)
    tfidf_nb_score = metrics.accuracy_score(y_test, tfidf_nb_pred)

    count_nb = MultinomialNB()
    count_nb.fit(count_train, y_train)
    count_nb_pred = count_nb.predict(count_test)
    count_nb_score = metrics.accuracy_score(y_test, count_nb_pred)

    #print('Naive Bayes Tdidf score: ', tfidf_nb_score)
    #print('Naive Bayes Count score: ', count_nb_score)

    # Calculate accuracy of model over testing data & confusion matrix
    print('\n*** Multinominal Naive Bayes Classifier ***')
    print('\n-- Tf-idf')
    accuracy(y_test, tfidf_nb_pred)
    dispConfusionMatrix(y_test, tfidf_nb_pred, classifierTfidf, nom)
    print('\n-- Count')
    accuracy(y_test, count_nb_pred)
    dispConfusionMatrix(y_test, count_nb_pred, classifierCount, nom)



   
###  Linear SVC Classifier  ###
def linearSVC(dataset):
    classifierSVC = 'Linear SVC Confusion Matrix'
    nom = 'SVC'

    # Create target
    y = dataset['label']

    # Divide data for training and testing (currently 80:20 - train:test)
    x_train, x_test, y_train, y_test = train_test_split(dataset['text'], y, test_size = 0.2, random_state = 11)

    # Pre-process data with CountVectorizer and TfidfVectorizor because ML algorithms only work with numerical data
    # CountVectorizer creates dictionary with occurrence number of tokens
    count_vectorizer = CountVectorizer(stop_words='english', min_df = 0.05, max_df = 0.9)
    count_train = count_vectorizer.fit_transform(x_train, y_train)
    count_test = count_vectorizer.transform(x_test)

    # TfidfVectorizer creates dictionary with tf-idf values of tokens
    # It determines the importance of a particular token, if it is common - value will be low, if it is rare - value will be high
    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 0.05, max_df = 0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train, y_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)

    # Create Linear SVM model with tf-idf 
    tfidf_svc = LinearSVC()
    tfidf_svc.fit(tfidf_train, y_train)
    tfidf_svc_pred = tfidf_svc.predict(tfidf_test)
    tfidf_svc_score = metrics.accuracy_score(y_test, tfidf_svc_pred)

    # Calculate accuracy of model over testing data & confusion matrix
    print('\n*** Linear SVC Classifier ***')
    accuracy(y_test, tfidf_svc_pred)
    dispConfusionMatrix(y_test, tfidf_svc_pred, classifierSVC, nom)


###########################################################################

#####  Main Program  #####


# Dataset source
dataFile = './kaggle-covid-news.csv'
#dataFile = './general-news.csv'
#dataFile = './covid-news.csv'
#dataFile = './general-WELFake.csv'

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
passAggrModel = passiveAggressive(x_train, x_test, y_train, y_test)
logicRegModel = logicRegression(x_train, x_test, y_train, y_test)
decTreeModel = decisionTree(x_train, x_test, y_train, y_test)
randForModel = randomForest(x_train, x_test, y_train, y_test)
naiveBayesModel = naiveBayes(data)
linearSVCModel = linearSVC(data)


##########################################################################

##  Query Tweet/headline/text for fake news determination on Covid

# News article input
news = 'covid is hoax'
print('\nNews article reads: ', news, '\n')

# Passive Aggressive Classifier result
result = passAggrModel.predict([news])
conf = passAggrModel.predict_proba([news])
print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
#print('Passive Aggressive result is: ', conf[0])
#print('Passive Aggressive confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')
#print("Confidence: ", round(conf[0][0]*100,2), "%")

# Logic Regression result
result = logicRegModel.predict([news])
conf = logicRegModel.predict_proba([news])
print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
#print('Logic Regression confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')

# Decision Tree Classifier result
result = decTreeModel.predict([news])
conf = decTreeModel.predict_proba([news])
print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Random Forest Classifier result
result = randForModel.predict([news])
conf = randForModel.predict_proba([news])
print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Naive Bayes Classifier result
#input = [news]
#vectorizor = CountVectorizer()
#vecNews = vectorizor.fit_transform(input)
#vecNews = vectorizor.transform([news]).toarray()
#vecNews = news.toarray()
#print(vecNews)
#naiveBayesModel.fit(vecNews).values
#result = naiveBayesModel.predict([news])
#conf = naiveBayesModel.predict_proba([news])
#print('Naive Bayes result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
#print('Naive Bayes result is: \t', result[0])

########################################

# News article input
news = 'washing your hands regularly is one of the best ways to prevent the spead of coronavirus'
print('\nNews article reads: ', news, '\n')

# Passive Aggressive Classifier result
result = passAggrModel.predict([news])
conf = passAggrModel.predict_proba([news])
print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Logic Regression result
result = logicRegModel.predict([news])
conf = logicRegModel.predict_proba([news])
print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Decision Tree Classifier result
result = decTreeModel.predict([news])
conf = decTreeModel.predict_proba([news])
print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Random Forest Classifier result
result = randForModel.predict([news])
conf = randForModel.predict_proba([news])
print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

########################################

# News article input
news = 'The novel coronavirus outbreak has spread to more than 150 countries or territories around the world'
print('\nNews article reads: ', news, '\n')

# Passive Aggressive Classifier result
result = passAggrModel.predict([news])
conf = passAggrModel.predict_proba([news])
print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Logic Regression result
result = logicRegModel.predict([news])
conf = logicRegModel.predict_proba([news])
print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Decision Tree Classifier result
result = decTreeModel.predict([news])
conf = decTreeModel.predict_proba([news])
print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Random Forest Classifier result
result = randForModel.predict([news])
conf = randForModel.predict_proba([news])
print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

########################################

# News article input
news = 'Coronavirus is only dangerous for old people'
print('\nNews article reads: ', news, '\n')

# Passive Aggressive Classifier result
result = passAggrModel.predict([news])
conf = passAggrModel.predict_proba([news])
print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Logic Regression result
result = logicRegModel.predict([news])
conf = logicRegModel.predict_proba([news])
print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Decision Tree Classifier result
result = decTreeModel.predict([news])
conf = decTreeModel.predict_proba([news])
print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Random Forest Classifier result
result = randForModel.predict([news])
conf = randForModel.predict_proba([news])
print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')


##########################################################################


##  Query Tweet/headline/text for fake news determination on general news

# News article input
news = 'NASA is installing internet on the moon'
print('\nNews article reads: ', news, '\n')

# Passive Aggressive Classifier result
result = passAggrModel.predict([news])
conf = passAggrModel.predict_proba([news])
print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
#print('Passive Aggressive result is: ', conf[0])
#print('Passive Aggressive confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')
#print("Confidence: ", round(conf[0][0]*100,2), "%")

# Logic Regression result
result = logicRegModel.predict([news])
conf = logicRegModel.predict_proba([news])
print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
#print('Logic Regression confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')

# Decision Tree Classifier result
result = decTreeModel.predict([news])
conf = decTreeModel.predict_proba([news])
print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Random Forest Classifier result
result = randForModel.predict([news])
conf = randForModel.predict_proba([news])
print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

########################################

# News article input
news = 'Spinach is taught how to send emails'
print('\nNews article reads: ', news, '\n')

# Passive Aggressive Classifier result
result = passAggrModel.predict([news])
conf = passAggrModel.predict_proba([news])
print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
#print('Passive Aggressive result is: ', conf[0])
#print('Passive Aggressive confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')
#print("Confidence: ", round(conf[0][0]*100,2), "%")

# Logic Regression result
result = logicRegModel.predict([news])
conf = logicRegModel.predict_proba([news])
print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
#print('Logic Regression confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')

# Decision Tree Classifier result
result = decTreeModel.predict([news])
conf = decTreeModel.predict_proba([news])
print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Random Forest Classifier result
result = randForModel.predict([news])
conf = randForModel.predict_proba([news])
print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

########################################

# News article input
news = 'Donald Trump is President'
print('\nNews article reads: ', news, '\n')

# Passive Aggressive Classifier result
result = passAggrModel.predict([news])
conf = passAggrModel.predict_proba([news])
print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
#print('Passive Aggressive result is: ', conf[0])
#print('Passive Aggressive confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')
#print("Confidence: ", round(conf[0][0]*100,2), "%")

# Logic Regression result
result = logicRegModel.predict([news])
conf = logicRegModel.predict_proba([news])
print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
#print('Logic Regression confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')

# Decision Tree Classifier result
result = decTreeModel.predict([news])
conf = decTreeModel.predict_proba([news])
print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Random Forest Classifier result
result = randForModel.predict([news])
conf = randForModel.predict_proba([news])
print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

########################################

# News article input
news = 'Donald Trump is a liar'
print('\nNews article reads: ', news, '\n')

# Passive Aggressive Classifier result
result = passAggrModel.predict([news])
conf = passAggrModel.predict_proba([news])
print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
#print('Passive Aggressive result is: ', conf[0])
#print('Passive Aggressive confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')
#print("Confidence: ", round(conf[0][0]*100,2), "%")

# Logic Regression result
result = logicRegModel.predict([news])
conf = logicRegModel.predict_proba([news])
print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
#print('Logic Regression confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')

# Decision Tree Classifier result
result = decTreeModel.predict([news])
conf = decTreeModel.predict_proba([news])
print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

# Random Forest Classifier result
result = randForModel.predict([news])
conf = randForModel.predict_proba([news])
print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

