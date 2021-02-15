# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
pr_data=pd.read_csv('mapreduce法最后PR值.csv',encoding='utf-8')
pr_sorted=pr_data.sort_values(axis=0,ascending=False,by=['PR'])
# #聚类得分
# print(pr_sorted)
X=np.array(pr_sorted['PR']).reshape(-1,1)#如何
# print(len(X))
# X=np.array(pr_sorted['PR'])
# print(X)
for i in range(2,20):
    y_pred = KMeans(n_clusters=i, random_state=9).fit_predict(X)
    print(calinski_harabasz_score(X, y_pred))
# y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
# print(calinski_harabasz_score(X, y_pred))
# print(pr_sorted)
