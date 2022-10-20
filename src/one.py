from sklearn import svm
from sklearn.ensemble import IsolationForest
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from clustering import *

faults = ['C2','C3','C4','C5','C6','C7']
sensors=['012','345','678','91011']
accum_percent = []
data_train=[]
data_test=[]
for fault in faults:
    # np.random.seed(10)
    b = np.zeros((4752, 300))
    b2 = np.zeros((2376, 300))
    for sensor in sensors:
        path = '../results/bigan/data_encoded_f_'+fault+'_'+sensor+'.pkl'
        with open(path, 'rb') as f:
            X_train, X_test = pickle.load(f)
        b=np.concatenate((b,X_train),axis=1)
        b2=np.concatenate((b2,X_test), axis=1)
    b=b[:,300:]
    b2=b2[:, 300:]

    mix = [i for i in range(len(b))]
    np.random.shuffle(mix)
    b = b[mix]
    b=b[1,:]

    data_train.append(b)
    data_test.append(b2)

num=1
train_x=np.array(data_train).reshape(1*len(faults),1200)
test_x=np.array(data_test).reshape(b2.shape[0]*len(faults),b2.shape[1])
train_y=np.array([0]*num+[1]*num+[2]*num+[3]*num+[4]*num+[5]*num).reshape(num*len(faults),)
test_y=np.array([0]*2376+[1]*2376+[2]*2376+[3]*2376+[4]*2376+[5]*2376).reshape(b2.shape[0]*len(faults),)
print(train_y.shape,train_x.shape)
print(test_y.shape,test_x.shape)
#
# # 3,Random Forest
cart_model = RandomForestClassifier(100)
cart_model.fit(train_x, train_y)
result = cart_model.score(test_x, test_y)
labels_pred = cart_model.predict(test_x)
acc=accuracy_score(test_y,labels_pred)
print(classification_report(test_y,labels_pred,output_dict=True))
print('the RF acc is ',acc)
# print(classification_report(test_y,labels_pred,output_dict=True))
print(confusion_matrix(test_y,labels_pred))
