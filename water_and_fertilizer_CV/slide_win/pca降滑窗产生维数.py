import pandas as pd
from sklearn.decomposition import PCA
import bisect

# l = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]


for i in range(1,20):
    outputfile='res_winsize'+str(i)+'.csv'
    data=pd.read_csv(outputfile)
    label=data['tick']
    data=data.drop(['tick'],axis=1)#默认是行，1为列
    pca = PCA()
    pca.fit(data)
    low_d = pca.transform(data)  # 用它来降低维度
    res=pca.explained_variance_ratio_
    res=sorted(res)
    print(res)
    res1=bisect.bisect(res, 0.00001)
    num=len(res)-res1
    pca = PCA(num)
    pca.fit(data)
    low_d = pca.transform(data)  # 用它来降低维度
    low_d=pd.DataFrame(low_d)
    low_d['tick']=list(label)
    low_d.to_csv('pca'+outputfile)  # 保存结果
    # pca.inverse_transform(low_d)  # 必要时可以用inverse_transform()函数来复原数据