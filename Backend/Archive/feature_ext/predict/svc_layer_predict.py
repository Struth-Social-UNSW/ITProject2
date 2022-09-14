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


featuresx = np.array([[1,2,3,4,5,6,7,8,9,10], [2,3,4,5,6,7,8,9,10,11], [3,4,5,6,7,8,9,10,11,12], [4,5,6,7,8,9,10,11,12,13]])
featuresy = np.array([1,0,0,1])

training(featuresx, featuresy)
