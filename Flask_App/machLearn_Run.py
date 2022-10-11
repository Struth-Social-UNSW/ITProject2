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
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import re


#import files
import dl_preproc_predict

# Import libraries from scikit learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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

# Import libraries from WordCloud
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
    print('\nDataset.head: \n', dataset.head())

    return dataset


###  Pre-process input  ###
def input_preprocess(input_string):

    processed_string = ''
 
    # Remove punctuation
    processed_string = input_string.lower()

    # Remove emojis
    filter_char = lambda c: ord(c) < 256
    processed_string = ''.join(filter(filter_char, processed_string))

     ## Removing URLs
    remurl = re.sub('http://\S+|https://\S+', '', processed_string)

    ## Removing Twitter Handles
    remhand = re.sub('@[^\s]+', '', remurl)

    ## Removing Hashtags (general)
    remhash4 = re.sub('#[^\s]+', '', remhand)
    
    ## Removing twitterurl tags
    remtwitterurl = remhash4.replace('twitterurl', '')
    
    ## Removing twitteruser tags
    remtwitteruser = remtwitterurl.replace('twitteruser', '')
    
    ## Removing rt tags
    remrt = remtwitteruser.replace('rt', '')

    # Remove stopwords
    stop = stopwords.words('english')
    processed_string = ' '.join([word for word in remrt.split() if word not in (stop)])

    return processed_string


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
    #f1Score = f1_score(y_test, predicted)
    #print("F1 Score: ", round(f1Score*100,2), "%")
    print("Precision:   measures the proportion of positively predicted labels that are acutualy correct.")
    print("Recall:      represents the model's ability to correctly predict the positives out of actual positives.")
    print("F1 Score:    represents the model score as a function of precision and recall score.")
    print("Recall:      represents the model's ability to correctly predict the positives out of actual positives.\n")
    print(metrics.classification_report(y_test, predicted))


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

    # Load the saved model
    DumpFilename = './SavedModelStates/finalized_passiveAggressive.sav'
    model = pickle.load(open(DumpFilename, 'rb'))

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
    
    # De-serialize data stream
    with open('model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    return model


###  Logistic Regression  ###
def logicRegression(x_train, x_test, y_train, y_test):

    # Load the saved model
    DumpFilename = './SavedModelStates/finalized_logicRegression.sav'
    model = pickle.load(open(DumpFilename, 'rb'))

    # Accuracy
    predicted = model.predict(x_test)

    # Calculate accuracy of model over testing data
    print('\n*** Logistic regression ***')
    accuracy(y_test, predicted)

    # Display confusion matrix
    dispConfusionMatrix(y_test, predicted)

    return model

    
###  Decision Tree Classifier  ###
def decisionTree(x_train, x_test, y_train, y_test):

    # Load the saved model
    DumpFilename = './SavedModelStates/finalized_decisionTree.sav'
    model = pickle.load(open(DumpFilename, 'rb'))

    # Accuracy
    predicted = model.predict(x_test)

    # Calculate accuracy of model over testing data
    print('\n*** Decision Tree Classifier ***')
    accuracy(y_test, predicted)

    # Display confusion matrix
    dispConfusionMatrix(y_test, predicted)

    return model


###  Random Forest Classifier  ###
def randomForest(x_train, x_test, y_train, y_test):
    
    # Load the saved model
    DumpFilename = './SavedModelStates/finalized_randomForest.sav'
    model = pickle.load(open(DumpFilename, 'rb'))

    # Accuracy
    predicted = model.predict(x_test)

    # Calculate accuracy of model over testing data
    print('\n*** Random Forest Classifier ***')
    accuracy(y_test, predicted)

    # Display confusion matrix
    dispConfusionMatrix(y_test, predicted)

    return model


###  Run input through the DL model and yield a prediction  ###
def DeepLearning(input):
    # tokenizer = AutoTokenizer.from_pretrained("bvrau/covid-twitter-bert-v2-struth") 
    pipe = pipeline("text-classification", model='bvrau/covid-twitter-bert-v2-struth')
    result = pipe(input)
    resultdict = result[0]
    label = resultdict['label']
    score = resultdict['score']
    MultipliedScore = (score*100)
    RoundedScore = round(MultipliedScore,2)
    # print("** Results **")
    # print("Determination: "+ label)
    # print("Certainty: "+str(score)) 
    DLResults = (label, (f'{str(RoundedScore)}%'))

    return DLResults




###########################################################################

###  Clean confidence scores and generate a usefull array  ###
def getConfidence(ConfidenceArray):
    confidenceMultiplied = [element * 100 for element in ConfidenceArray]
    confidenceRounded = [round(element,2) for element in confidenceMultiplied]
    confidenceMax = max(confidenceRounded)
    return confidenceMax


#####  Main Program  #####
def Main(InputArray, Mode):
    # Dataset source
    # dataFile = './general-WELFake.csv'
    # dataFile = './kaggle-covid-news.csv'
    # dataFile = './general-news.csv'

    # Load and read dataset
    # data = read(dataFile)

    # # Conduct preprocessing of dataset
    # data = preprocess(data)

    # print(type(data))
    data = pd.read_pickle("./SavedModelStates/preprocessed_data.pkl")

    # Create images/plots of data
    # fakeCloud(data)
    # realCloud(data)
    # countWords(data)


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

        #Preprocess user input
        print(f'Raw input: {IndividualInputText}')
        preprocessedIndividualTextInput  = input_preprocess(IndividualInputText)
        print(f'Preprocessed raw input: {preprocessedIndividualTextInput}')

        PassiveAggressiveClassifierResult = passAggrModel.predict([preprocessedIndividualTextInput])
        LogisiticRegressionClassifierResult = logicRegModel.predict([preprocessedIndividualTextInput])
        DecisionTreeClassifierResult = decTreeModel.predict([preprocessedIndividualTextInput])
        RandomForestClassifierResult = randForModel.predict([preprocessedIndividualTextInput])

        passAggrModel_confidenceArray = passAggrModel.predict_proba([preprocessedIndividualTextInput])[0]
        # print(passAggrModel_confidenceArray) #debug
        passAggrModel_confidence = str(f'{getConfidence(passAggrModel_confidenceArray)}%')
        # print(passAggrModel_confidence)

        logicRegModel_confidenceArray = logicRegModel.predict_proba([preprocessedIndividualTextInput])[0]
        # print(logicRegModel_confidenceArray) #debug
        logicRegModel_confidence = str(f'{getConfidence(logicRegModel_confidenceArray)}%')
        # print(logicRegModel_confidence)

        decTreeModel_confidenceArray = decTreeModel.predict_proba([preprocessedIndividualTextInput])[0]
        # print(decTreeModel_confidenceArray) #debug
        decTreeModel_confidence = str(f'{getConfidence(decTreeModel_confidenceArray)}%')
        # print(decTreeModel_confidence)

        randForModel_confidenceArray = randForModel.predict_proba([preprocessedIndividualTextInput])[0]
        # print(randForModel_confidenceArray) #debug
        randForModel_confidence = str(f'{getConfidence(randForModel_confidenceArray)}%')
        # print(randForModel_confidence)

        if Mode == 'ML':
            
            ResultsTupple = [IndividualInputText, PassiveAggressiveClassifierResult[0], LogisiticRegressionClassifierResult[0], DecisionTreeClassifierResult[0], RandomForestClassifierResult[0], 
                            (passAggrModel_confidence, logicRegModel_confidence, decTreeModel_confidence, randForModel_confidence)]
            Results.append(ResultsTupple)

        elif Mode == 'MLDL':

            #Perform DL prediction 
            #MUST RUN: python -m spacy download en_core_web_sm
            dl_preproccessed_input = dl_preproc_predict.preprocmain(IndividualInputText)
            DeepLearningResults = DeepLearning(dl_preproccessed_input)

            ResultsTupple = [IndividualInputText, PassiveAggressiveClassifierResult[0], LogisiticRegressionClassifierResult[0], DecisionTreeClassifierResult[0], RandomForestClassifierResult[0], 
                            (passAggrModel_confidence, logicRegModel_confidence, decTreeModel_confidence, randForModel_confidence), DeepLearningResults]
            Results.append(ResultsTupple)
    
    print(Results) #Debug
    return Results


##### TEST HARNESS FOR MAIN METHOD #####
if __name__ == '__main__':
    Main(['Covid is a hoax'], 'ML')
    print('\n\n################################################################\n\n\n')
    Main(['Covid is a hoax'], 'MLDL')


    


    ##  Query Tweet/headline/text for fake news determination on Covid

    # News article input
    #news = 'covid is hoax'
    #print('\nNews article reads: ', news, '\n')

    # Passive Aggressive Classifier result
    #result = passAggrModel.predict([news])
    #conf = passAggrModel.predict_proba([news])
    #print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
    #print('Passive Aggressive result is: ', conf[0])
    #print('Passive Aggressive confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')
    #print("Confidence: ", round(conf[0][0]*100,2), "%")

    # Logic Regression result
    #result = logicRegModel.predict([news])
    #conf = logicRegModel.predict_proba([news])
    #print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
    #print('Logic Regression confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')

    # Decision Tree Classifier result
    #result = decTreeModel.predict([news])
    #conf = decTreeModel.predict_proba([news])
    #print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Random Forest Classifier result
    #result = randForModel.predict([news])
    #conf = randForModel.predict_proba([news])
    #print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')


    # News article input
    #news = 'washing your hands regularly is one of the best ways to prevent the spead of coronavirus'
    #print('\nNews article reads: ', news, '\n')

    # Passive Aggressive Classifier result
    #result = passAggrModel.predict([news])
    #conf = passAggrModel.predict_proba([news])
    #print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Logic Regression result
    #result = logicRegModel.predict([news])
    #conf = logicRegModel.predict_proba([news])
    #print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Decision Tree Classifier result
    #result = decTreeModel.predict([news])
    #conf = decTreeModel.predict_proba([news])
    #print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Random Forest Classifier result
    #result = randForModel.predict([news])
    #conf = randForModel.predict_proba([news])
    #print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')


    # News article input
    #news = 'The novel coronavirus outbreak has spread to more than 150 countries or territories around the world'
    #print('\nNews article reads: ', news, '\n')

    # Passive Aggressive Classifier result
    #result = passAggrModel.predict([news])
    #conf = passAggrModel.predict_proba([news])
    #print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Logic Regression result
    #result = logicRegModel.predict([news])
    #conf = logicRegModel.predict_proba([news])
    #print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Decision Tree Classifier result
    #result = decTreeModel.predict([news])
    #conf = decTreeModel.predict_proba([news])
    #print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Random Forest Classifier result
    #result = randForModel.predict([news])
    #conf = randForModel.predict_proba([news])
    #print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')


    # News article input
    #news = 'Coronavirus is only dangerous for old people'
    #print('\nNews article reads: ', news, '\n')

    # Passive Aggressive Classifier result
    #result = passAggrModel.predict([news])
    #conf = passAggrModel.predict_proba([news])
    #print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Logic Regression result
    #result = logicRegModel.predict([news])
    #conf = logicRegModel.predict_proba([news])
    #print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Decision Tree Classifier result
    #result = decTreeModel.predict([news])
    #conf = decTreeModel.predict_proba([news])
    #print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Random Forest Classifier result
    #result = randForModel.predict([news])
    #conf = randForModel.predict_proba([news])
    #print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')




    ##  Query Tweet/headline/text for fake news determination on general news

    # News article input
    #news = 'NASA is installing internet on the moon'
    #print('\nNews article reads: ', news, '\n')

    # Passive Aggressive Classifier result
    #result = passAggrModel.predict([news])
    #conf = passAggrModel.predict_proba([news])
    #print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
    #print('Passive Aggressive result is: ', conf[0])
    #print('Passive Aggressive confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')
    #print("Confidence: ", round(conf[0][0]*100,2), "%")

    # Logic Regression result
    #result = logicRegModel.predict([news])
    #conf = logicRegModel.predict_proba([news])
    #print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
    #print('Logic Regression confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')

    # Decision Tree Classifier result
    #result = decTreeModel.predict([news])
    #conf = decTreeModel.predict_proba([news])
    #print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Random Forest Classifier result
    #result = randForModel.predict([news])
    #conf = randForModel.predict_proba([news])
    #print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')


    # News article input
    #news = 'Spinach is taught how to send emails'
    #print('\nNews article reads: ', news, '\n')

    # Passive Aggressive Classifier result
    #result = passAggrModel.predict([news])
    #conf = passAggrModel.predict_proba([news])
    #print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
    #print('Passive Aggressive result is: ', conf[0])
    #print('Passive Aggressive confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')
    #print("Confidence: ", round(conf[0][0]*100,2), "%")

    # Logic Regression result
    #result = logicRegModel.predict([news])
    #conf = logicRegModel.predict_proba([news])
    #print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
    #print('Logic Regression confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')

    # Decision Tree Classifier result
    #result = decTreeModel.predict([news])
    #conf = decTreeModel.predict_proba([news])
    #print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Random Forest Classifier result
    #result = randForModel.predict([news])
    #conf = randForModel.predict_proba([news])
    #print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')


    # News article input
    #news = 'Donald Trump is President'
    #print('\nNews article reads: ', news, '\n')

    # Passive Aggressive Classifier result
    #result = passAggrModel.predict([news])
    #conf = passAggrModel.predict_proba([news])
    #print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
    #print('Passive Aggressive result is: ', conf[0])
    #print('Passive Aggressive confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')
    #print("Confidence: ", round(conf[0][0]*100,2), "%")

    # Logic Regression result
    #result = logicRegModel.predict([news])
    #conf = logicRegModel.predict_proba([news])
    #print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
    #print('Logic Regression confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')

    # Decision Tree Classifier result
    #result = decTreeModel.predict([news])
    #conf = decTreeModel.predict_proba([news])
    #print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Random Forest Classifier result
    #result = randForModel.predict([news])
    #conf = randForModel.predict_proba([news])
    #print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')


    # News article input
    #news = 'Donald Trump is a liar'
    #print('\nNews article reads: ', news, '\n')

    # Passive Aggressive Classifier result
    #result = passAggrModel.predict([news])
    #conf = passAggrModel.predict_proba([news])
    #print('Passive Aggressive result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
    #print('Passive Aggressive result is: ', conf[0])
    #print('Passive Aggressive confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')
    #print("Confidence: ", round(conf[0][0]*100,2), "%")

    # Logic Regression result
    #result = logicRegModel.predict([news])
    #conf = logicRegModel.predict_proba([news])
    #print('Logic Regression result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')
    #print('Logic Regression confidence rating of: ', round(conf[0][0]*100,2), '% fake,', round(conf[0][1]*100,2), '% real')

    # Decision Tree Classifier result
    #result = decTreeModel.predict([news])
    #conf = decTreeModel.predict_proba([news])
    #print('Decision Tree result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

    # Random Forest Classifier result
    #result = randForModel.predict([news])
    #conf = randForModel.predict_proba([news])
    #print('Random Forest result is: \t', result[0], ' - with confidence rating of  ', round(conf[0][0]*100,2), '% fake, ', round(conf[0][1]*100,2), '% real')

