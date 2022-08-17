""" svc_layer.py: 
    
    This program is the first of the ML layers to receive the feature extraction data from dl_model.py and
    begin the classification tasks.
"""

__author__      = "Breydon Verryt-Reid"
__date__        = "15 Aug 22"
__Version__     = 1.0


import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

def training(features, labels):
    X = features
    y = labels
    print("Beginnning Fit Task")
    clf.fit(X, y)
    
def testing(feature, label):
    result = clf.predict(feature)
    if feature == label:
        print('Success')
    else:
        print("Failed")




def mlmodelmain():
    featuresx = np.array([])
    featuresy = np.array([])
    
    vectors = pd.read_csv('./dl_data.csv')
    
    for index in vectors.index:
        print([vectors['vector'][index]])
        np.insert(featuresx, 0, ([vectors['vector'][index]]))
        print(featuresx)
    
    training(featuresx, featuresy)
