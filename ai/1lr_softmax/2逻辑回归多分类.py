#encoding=utf8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
# [:, :2]：所有行，0、1 列，不包含 2 列；
X = iris.data[:145,:]
y = iris.target[:145]
Z = iris.data[145:,:]
Zy=iris.target[145:]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
#LogisticRegression() 默认使用 OvR
log_reg_ovr = LogisticRegression()
log_reg_ovr.fit(X_train, y_train)
score=log_reg_ovr.score(X_test, y_test)
print(score)
print(log_reg_ovr.predict_proba(Z))
print(log_reg_ovr.predict(Z))
# ‘multinomial‘：指 OvO 方法；
log_reg_ovo = LogisticRegression(multi_class='multinomial', solver='newton-cg')
log_reg_ovo.fit(X_train, y_train)
score=log_reg_ovo.score(X_test, y_test)
print(log_reg_ovo.predict_proba(Z))
print(log_reg_ovo.predict(Z))
print(score)

