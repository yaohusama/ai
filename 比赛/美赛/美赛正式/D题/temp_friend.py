from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import cross_val_score
from tqdm import trange,tqdm
from matplotlib.pyplot import MultipleLocator#根据拥挤程度调节试图效果
import seaborn as sns

influence_data=pd.read_csv('influence_data.csv')

genre_dict={}
trans_dict={'Pop/Rock':1,'R&B;':2,'Country':3,'Jazz':4,'Vocal':5,'Blues':6,'Folk':7,'Reggae':8,'Electronic':9,'Latin':10,
            'International':11,'Religious':12,'Stage & Screen':13,'Comedy/Spoken':14,'Classical':15,'New Age':16,'Avant-Garde':17,
            'Easy Listening':18,"Children's":19,'Unknown':20}
factor=['danceability','energy','valence','tempo','loudness','mode','key',
             'acousticness','instrumentalness','liveness','speechiness','explicit','duration_ms']
genre_list=['Pop/Rock','R&B;','Country','Jazz','Vocal','Blues','Folk','Reggae','Electronic','Latin','International','Religious',
       'Stage & Screen','Comedy/Spoken','Classical','New Age','Avant-Garde','Easy Listening',"Children's",'Unknown']

influence_len=len(influence_data)
genre_data=np.array(influence_data)
for i in range(influence_len):
    if genre_data[i][0] not in genre_dict.keys():
        genre_dict[genre_data[i][0]]=genre_data[i][2]
    if genre_data[i][4] not in genre_dict.keys():
        genre_dict[genre_data[i][4]]=genre_data[i][6]

music_data=np.array(pd.read_csv('full_music_data.csv'))

music_len=len(music_data)
for i in trange(music_len):
    music_data[i,1]=music_data[i,1].strip('[').strip(']')
    if ',' not in music_data[i,1]:
        music_data[i,1]=int(music_data[i,1])
    else:
        music_data[i,1]=music_data[i,1].split(',')
        music_data[i,1]=[int(j) for j in music_data[i,1]]
        len_=len(music_data[i,1])-1
        for j in range(len_):
            music_data=np.concatenate((music_data,music_data[i].reshape(1,-1)),axis=0)#默认axis=0即行
            music_data[-1,1]=music_data[i,1][-1]
            music_data[i,1].pop(-1)
        music_data[i,1]=music_data[i,1][0]
        
# np.savetxt('music_data.csv', music_data)

y=[]
delete=[]
music_len=len(music_data)#103487
for i in range(music_len):
    if music_data[i,1] in genre_dict.keys():
        genre=genre_dict[music_data[i,1]]
        label=trans_dict[genre]
        y.append(label)
    else:
        delete.append(i)
music_data = np.delete(music_data,delete, axis = 0)#101092

X=music_data[:,2:15]
X=MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
#X = (X-X.mean())/(X.std())  

y=np.array(y)
'''
data=pd.DataFrame(X,columns=['danceability','energy','valence','tempo','loudness','mode','key',
             'acousticness','instrumentalness','liveness','speechiness','explicit','duration_ms','popularity'])
data['label']=pd.DataFrame(y)


music_genre=[]
for i in range(20):
    #music_genre.append(data[data['label']==i+1])
    p=data[data['label']==i+1].sort_values(by='popularity',ascending=False)
    p=np.array(p.drop(columns='popularity'))
    len_=int(len(p)/5)
    music_genre.append(pd.DataFrame(p[:len_,:],columns=['danceability','energy','valence','tempo','loudness','mode','key',
             'acousticness','instrumentalness','liveness','speechiness','explicit','duration_ms','label']))
    
X_train=music_genre[0]
for i in range(19):
    X_train=pd.concat([X_train,music_genre[i+1]])
    
y_train=np.array(X_train['label'])
X_train=np.array(X_train.drop(columns='label'))
'''

'''
#热力图
data=pd.DataFrame(X,columns=factor)
data['label']=pd.DataFrame(y)

music_genre=[]
for i in range(20):
    music_genre.append(data[data['label']==i+1])
    graphs=music_genre[i].drop(columns=['label']).corr()
    plt.subplots(figsize=(9, 9))
    sns.heatmap(graphs, annot=True, vmax=1, square=True, cmap="Reds")
    plt.title('Pearson correlation coefficient in the genre of '+genre_list[i])
    plt.show()
'''

cv_score=[]
k_range=range(1,20)
for i in tqdm(k_range): 
    clf = KNeighborsClassifier(n_neighbors=i)
    scores=cross_val_score(clf,X,y.ravel(),cv=10)
    cv_score.append(scores.mean())

plt.plot(k_range,cv_score,color="red",label="accuracy-n_neighbors curve") 
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')

x_major_locator=MultipleLocator(1)#xmajorLocator =MultipleLocator(20)#将x主刻度标签设置为20的倍数
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.set_title('relationship between accuracy and number of n_neighbors',fontsize=14)

plt.grid()
plt.legend(loc='best') 
plt.show()


'''
pca=PCA(n_components=2)
X=pca.fit_transform(X)
plt.scatter(X[:,0], X[:,1], s=4)
'''


