from sklearn import svm#SVM和朴素贝叶斯模板
from sklearn import datasets	#自带数据集
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.utils import shuffle


train_data_='train_data.csv'
train_label_='train_label.csv'
test_data_='test_data.csv'
test_label_='test_label.csv'

train_data=np.array(pd.read_csv(train_data_,header=None))
train_label=np.array(pd.read_csv(train_label_,header=None))
test_data=np.array(pd.read_csv(test_data_,header=None))
test_label=np.array(pd.read_csv(test_label_,header=None))

'''
X,y = datasets.load_breast_cancer(return_X_y=True)
df_X=pd.DataFrame(X)
df_y=pd.DataFrame(y)
df_X['y']=df_y
df=shuffle(df_X)
df_y=df['y']
df=df.drop(columns='y')

X=np.array(df)
y=np.array(df_y)

train_data=X[:400]
train_label=y[:400]
test_data=X[400:]
test_label=y[400:]'''


gnb=GaussianNB()

start=time.time()
gnb.fit(train_data,train_label.ravel())#dataframe是没有Ravel，array需要Ravel
end=time.time()
print('naive bayes分类用时',end-start,'秒')

start=time.time()
training_score=gnb.score(train_data,train_label)#平均精度
end=time.time()
print("训练集得分：",training_score,'用时：',end-start,'秒')

start=time.time()
training_score=gnb.score(test_data,test_label)
end=time.time()
print("测试集得分：",training_score,'用时：',end-start,'秒')

# classifier=svm.SVC(C=2,kernel='linear',gamma=10,decision_function_shape='ovr') # ovr:一对多策略
#
# start=time.time()
# classifier.fit(train_data,train_label.ravel())
# end=time.time()
# print('svm分类用时',end-start,'秒')
#
# start=time.time()
# training_score=classifier.score(train_data,train_label)
# end=time.time()
# print("训练集得分：",training_score,'用时：',end-start,'秒')
#
# start=time.time()
# training_score=classifier.score(test_data,test_label)
# end=time.time()
# print("测试集得分：",training_score,'用时：',end-start,'秒')
