from sklearn import svm
from sklearn.ensemble import IsolationForest
import pickle
import numpy as np
from clustering import *

classifier = 'SVM'
sensors=['012','345','678','91011']
b1=np.zeros((2376,300))
b2=np.zeros((2376,300))
for sensor in sensors:

    path_train = '../results/dcae/data_encoded_f_C0_'+sensor+'.pkl'
    with open(path_train,'rb') as f:
        _,X_test1= pickle.load(f)
    b1 = np.concatenate((b1, X_test1), axis=1)

    path_train2 = '../results/dcae/data_encoded_f_C2_' + sensor + '.pkl'
    with open(path_train2, 'rb') as f:
        _,X_test2= pickle.load(f)
    b2 = np.concatenate((b2, X_test1), axis=1)

C0_C5=np.vstack((b1[:,300:],b2[:,300:]))
with open('../results/dcae/data_encoded_f_C0_C2.pkl','wb') as f:
    pickle.dump(C0_C5,f,pickle.HIGHEST_PROTOCOL)



