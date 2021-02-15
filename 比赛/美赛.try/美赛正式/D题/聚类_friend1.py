import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import calinski_harabasz_score
from pandas.plotting import scatter_matrix

data_by_artist = pd.read_csv('data_by_artist.csv')
data_by_year = pd.read_csv('data_by_year.csv')
full_music_data = pd.read_csv('full_music_data.csv')
# influence_data = pd.read_csv('data/influence_data.csv')

full_music_data['artist_names'] = full_music_data['artist_names'].map(lambda x: x.replace('["', '').replace('"]',''))
full_music_data['artists_id'] = full_music_data['artists_id'].map(lambda x: x.replace('[', '').replace(']', ''))

x = ['danceability', 'energy', 'valence','tempo', 'loudness', 'mode', 'key', 'acousticness', 'instrumentalness',
 'liveness', 'speechiness', 'explicit', 'duration_ms','year', 'popularity']
y = 'popularity'

means = full_music_data.describe().loc['mean', :]
stds = full_music_data.describe().loc['std', :]
pearson = []
music_data=full_music_data[x]
popularity=full_music_data['popularity']-means['popularity']

corr=music_data.corr()

# pearson
for i in x:
    fenzi=sum((music_data[i]-means[i])*popularity)/len(music_data[i])
    fenmu=(stds[i]*stds['popularity'])
    #print(i,fenzi/fenmu)
    
co1=corr[corr['popularity']>0.15]['popularity']
co2=corr[corr['popularity']<-0.15]['popularity']
co1=co1.append(co2)

trans_dict={'Pop/Rock':1,'R&B;':2,'Country':3,'Jazz':4,'Vocal':5,'Blues':6,'Folk':7,'Reggae':8,'Electronic':9,'Latin':10,
            'International':11,'Religious':12,'Stage & Screen':13,'Comedy/Spoken':14,'Classical':15,'New Age':16,'Avant-Garde':17,
            'Easy Listening':18,"Children's":19,'Unknown':20}

same_music_data = pd.read_excel('same_music_data.xlsx')
music_data=same_music_data
X = music_data[co1.index]
X = (X-X.min())/(X.max()-X.min())

label=music_data['genre'].map(trans_dict)


#X=MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

#聚类成两簇
km2 = KMeans(n_clusters=2).fit(X)
cluster_centers_2 = km2.cluster_centers_

music_data['cluster2'] = km2.labels_
music_data.sort_values('cluster2')#按指定列排序

cluster_centre=music_data["cluster2"]

centers = music_data.groupby("cluster2").mean().reset_index()#数据清洗时，会将带空值的行删除，此时DataFrame或Series类型的数据不再是连续的索引，可以使用reset_index()重置索引。

plt.rcParams['font.size'] = 18
colors = np.array(['red', 'blue'])
plt.scatter(music_data["loudness"], music_data["tempo"],c=colors[cluster_centre],s=8)
plt.scatter(centers.loudness, centers.tempo, linewidths=3, marker='+', s=300, c='black')#S是点的大小

plt.xlabel("loudness")
plt.ylabel("tempo")

scatter_matrix(X,s=32, alpha=1, c=colors[music_data["cluster2"]], figsize=(25,25))#使用散点图矩阵图，可以两两发现特征之间的联系
plt.suptitle("music data divided into 2 centroids",fontsize=50)

scatter_matrix(X,s=32, alpha=1, c=label, figsize=(25,25))
plt.suptitle("music data divided by genres",fontsize=50)#绘制总标题
plt.show()
'''
pca=PCA(n_components=2)
X=pca.fit_transform(X)
plt.scatter(X[:,0], X[:,1], s=4, c=label)
plt.show()
'''
'''
#聚类得分
y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
print(calinski_harabasz_score(X, y_pred))

y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
print(calinski_harabasz_score(X, y_pred))
'''
'''
km2 = KMeans(n_clusters=2).fit(X)
music_data['cluster2'] = km2.labels_
music_data.sort_values('cluster2')'''