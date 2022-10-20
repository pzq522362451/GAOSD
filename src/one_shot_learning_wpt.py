from sklearn import svm
from sklearn.ensemble import IsolationForest
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from clustering import *

faults = ['C4','C7','C5','C2']
sensors=['678','91011']
accum_percent = []
data_train=[]
data_test=[]
for fault in faults:
    np.random.seed(55)
    b = np.zeros((4752, 128))
    b2 = np.zeros((2376, 128))
    for sensor in sensors:
        path = '../results/wpt/data_encoded_f_'+fault+'_'+sensor+'.pkl'
        with open(path, 'rb') as f:
            X_train, X_test = pickle.load(f)
        b=np.concatenate((b,X_train),axis=1)
        b2=np.concatenate((b2,X_test), axis=1)
    b=b[:,128:]
    b2=b2[:, 128:]

    mix = [i for i in range(len(b))]
    np.random.shuffle(mix)
    b = b[mix]
    b=b[0:1,:]# 136 sae

    data_train.append(b)
    data_test.append(b2)

num=1
train_x=np.array(data_train).reshape(b.shape[0]*len(faults),b.shape[1])
test_x=np.array(data_test).reshape(b2.shape[0]*len(faults),b2.shape[1])
train_y=np.array([0]*num+[1]*num+[2]*num+[3]*num).reshape(num*len(faults),)
test_y=np.array([0]*2376+[1]*2376+[2]*2376+[3]*2376).reshape(b2.shape[0]*len(faults),)
print(train_y.shape,train_x.shape)
print(test_y.shape,test_x.shape)

# # 3,Random Forest
cart_model = RandomForestClassifier(100)
cart_model.fit(train_x, train_y)
result = cart_model.score(test_x, test_y)
labels_pred = cart_model.predict(test_x)
acc=accuracy_score(test_y,labels_pred)
# print(classification_report(test_y,labels_pred,output_dict=True))
print('the RF acc is ',acc)

print(classification_report(test_y,labels_pred,output_dict=True))
# print(confusion_matrix(test_y,labels_pred))
# t_sne(x=test_x,y=test_y,n_class=4,savename='../results/tsne_wpt2.png')
