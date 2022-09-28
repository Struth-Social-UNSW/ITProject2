# This program reads data, conducts ML training and testing, outputing an accuracy level and confusion matrix.
# 
# NOTE: The dataset used must contain real/fake labelling
#
#
# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

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
    dataset['text'] = dataset['text'].apply(lambda x: x.lower())
    #print('\nDataset.head: \n', dataset.head())

    # Remove punctuation
    dataset['text'] = dataset['text'].apply(punctuation_removal)
    #print('\nDataset.head: \n', dataset.head())

    # Remove stopwords
    stop = stopwords.words('english')
    dataset['text'] = dataset['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    # Print first 5 rows of dataset after pre-processing
    print('\nDataset.head: \n', dataset.head())

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
    plt.savefig('static/img_fakeCloud.jpg')

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
    plt.savefig('static/img_realCloud.jpg')

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
    plt.savefig('static/img_freqWords.jpg')

    # Displays frequent words table to screen
    #plt.show()
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


###  Confustion matrix  ###
def plotConfusionMatrix(cm, classes,
                            normalize = False,
                            title = 'Confusion matrix', 
                            cmap = plt.cm.Blues):
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
    plt.savefig('static/img_confMatrix.jpg')

    # Displays confusion matrix to the screen
    # plt.show()
    plt.close()

###   Display Confustion Matrix  ###
def dispConfusionMatrix(y_test, predicted):
    print(metrics.confusion_matrix(y_test, predicted))
    cm = metrics.confusion_matrix(y_test, predicted)
    plotConfusionMatrix(cm, classes=['Fake', 'True'])  ## Uncomment this line to save/display confusion matrix


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
    dispConfusionMatrix(y_test, predicted)

    # Pipeline utility function to train and transform data to text data
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words = 'english')), ('nbmodel', MultinomialNB())])
    pipeline.fit(x_train, y_train)

    # Pickling is process where object heirarchy is converted into a byte stream
    # Serialize an object hierarchy
    with open('model.pkl', 'wb') as handle:
        pickle.dump(pipeline, handle, protocol = pickle.HIGHEST_PROTOCOL)

    #Save Finalised model state from training into respective file in SavedModelStates directory
    DumpFilename = './SavedModelStates/finalized_passiveAggressive.sav'
    pickle.dump(model, open(DumpFilename, 'wb'))

    # De-serialize data stream
    with open('model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    return model


###  Logistic Regression  ###
def logicRegression(x_train, x_test, y_train, y_test):
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
    dispConfusionMatrix(y_test, predicted)

    #Save Finalised model state from training into respective file in SavedModelStates directory
    DumpFilename = './SavedModelStates/finalized_logicRegression.sav'
    pickle.dump(model, open(DumpFilename, 'wb'))

    return model

    
###  Decision Tree Classifier  ###
def decisionTree(x_train, x_test, y_train, y_test):
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
    dispConfusionMatrix(y_test, predicted)

    #Save Finalised model state from training into respective file in SavedModelStates directory
    DumpFilename = './SavedModelStates/finalized_decisionTree.sav'
    pickle.dump(model, open(DumpFilename, 'wb'))

    return model


###  Random Forest Classifier  ###
def randomForest(x_train, x_test, y_train, y_test):
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
    dispConfusionMatrix(y_test, predicted)

    #Save Finalised model state from training into respective file in SavedModelStates directory
    DumpFilename = './SavedModelStates/finalized_randomForest.sav'
    pickle.dump(model, open(DumpFilename, 'wb'))

    return model



###########################################################################

#####  Main Program  #####
def Main(InputArray):
    # Dataset source
    # dataFile = './general-WELFake.csv'
    dataFile = './kaggle-covid-news.csv'
    # dataFile = './general-news.csv'

    # Load and read dataset
    data = read(dataFile)

    # Conduct preprocessing of dataset
    data = preprocess(data)

    #Save preprocessed data to pickle file
    data.to_pickle("./SavedModelStates/preprocessed_data.pkl")

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



    ##  Query Tweet/headline/text for fake news determination
    Results = []
    for IndividualInputText in InputArray:

        PassiveAggressiveClassifierResult = passAggrModel.predict([IndividualInputText])
        LogisiticRegressionClassifierResult = logicRegModel.predict([IndividualInputText])
        DecisionTreeClassifierResult = decTreeModel.predict([IndividualInputText])
        RandomForestClassifierResult = randForModel.predict([IndividualInputText])

        ResultsTupple = [IndividualInputText, PassiveAggressiveClassifierResult[0], LogisiticRegressionClassifierResult[0], DecisionTreeClassifierResult[0], RandomForestClassifierResult[0]]
        Results.append(ResultsTupple)
    
    print(Results) #Debug
    return Results


##### TEST HARNESS FOR MAIN METHOD #####
if __name__ == '__main__':
    Main(['Covid is a hoax'])

    print('\nModel States Successfully Saved!')
    print('Preprocessed Dataset Successfully Saved!')


    
