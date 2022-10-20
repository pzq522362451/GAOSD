from sklearn import svm
from sklearn.ensemble import IsolationForest
import pickle
import numpy as np
from clustering import *

classifier = 'SVM'
faults=['C0','C5','C2','C3','C4','C6','C7','C8']
sensors=['012','345','678','91011']

for fault in faults:
    b = np.zeros((2376, 300))
    for sensor in sensors:
        path_train = '../results/dcae/data_encoded_f_'+fault+'_'+sensor+'.pkl'
        with open(path_train,'rb') as f:
            _,X_test= pickle.load(f)
        b = np.concatenate((b, X_test), axis=1)
    Class=b[:,300:]
    with open('../results/dcae/data_encoded_f_'+fault+'.pkl','wb') as f:
        pickle.dump(Class,f,pickle.HIGHEST_PROTOCOL)



