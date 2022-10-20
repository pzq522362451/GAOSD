from sklearn import svm
from sklearn.ensemble import IsolationForest
import pickle
import numpy as np

# 读取P数据
path_train = '../results/data_encoded_f_P.pkl'
classifier = 'RF'
with open(path_train,'rb') as f:
    X_train, _ = pickle.load(f)
media = X_train.mean(axis=0)
des = X_train.std(axis=0)
X_train = (X_train - media) / des

#进行iforest训练
clf = IsolationForest(behaviour='new', max_samples=.7, max_features=1., n_jobs=-1, bootstrap=False, contamination=0,n_estimators=500)
clf.fit(X_train)

#依次送入每类故障数据
path_test = '../results/data_encoded_f_'+fault+'.pkl'
with open(path_test, 'rb') as f:
    _, X_test = pickle.load(f)
X_test = (X_test - media) / des
y_pred_test = clf.predict(X_test)

#进行比较
if fault == 'P':
    n_error_test = y_pred_test[y_pred_test == -1].size
else:
    n_error_test = y_pred_test[y_pred_test == 1].size


#输出精度
percent = n_error_test/len(X_test)
print('% test errors condition',fault,':', percent)
accum_percent.append(1-percent)
print('Avg. Accuracy:', np.array(accum_percent).mean())