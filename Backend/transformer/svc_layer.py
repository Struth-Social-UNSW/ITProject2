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


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)