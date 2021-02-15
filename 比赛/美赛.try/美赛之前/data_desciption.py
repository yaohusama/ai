from __future__ import print_function
from sklearn.decomposition import PCA
import pandas as pd



train_normalize=pd.read_csv('train_data.csv',encoding='utf-8')
# train_normalize=(train_normalize-train_normalize.mean())/(train_normalize.std())
train_normalize.columns=range(118)
statistics = train_normalize.describe()  # 保存基本统计量

statistics.loc['range'] = statistics.loc['max']-statistics.loc['min']  # 极差
statistics.loc['var'] = statistics.loc['std']/statistics.loc['mean']  # 变异系数
statistics.loc['dis'] = statistics.loc['75%']-statistics.loc['25%']  # 四分位数间距

# print(statistics)

mat=train_normalize.corr()
strong_part=[]
mat.columns=range(118)
for i in range(118):
    for j in range(118):
        if mat.loc[i,j]>=0.8:
            strong_part.append((i,j))
print(strong_part)
del_part=[]
for i in strong_part:
    if i[0]!=i[1]:
        del_part.append(i[0])
train_normalize=train_normalize.drop(columns=del_part)

outputfile='PCA_process.csv'
pca = PCA()
pca.fit(train_normalize)
print(pca.components_ ,pca.explained_variance_ratio_,end=r'\n')
low_d = pca.transform(train_normalize)  # 用它来降低维度
pd.DataFrame(low_d).to_csv(outputfile)  # 保存结果
original_data=pca.inverse_transform(low_d)#原数据
