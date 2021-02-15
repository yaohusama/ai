import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


full_music_data = pd.read_csv('full_music_data.csv')
artist_data=pd.read_csv('data_by_artist.csv')
influence_data = pd.read_csv('influence_data.csv')


def influence_score_name(name):
    followers=np.array(influence_data.loc[influence_data['influencer_name']==name,:]['follower_name'])

    influenced = pd.DataFrame(columns=artist_data.columns)

    for follower in followers:    
        influenced=pd.concat([influenced,artist_data.loc[artist_data['artist_name'] == follower, :]])
    
    data=['danceability','energy','valence','tempo','loudness','mode','key','acousticness',
          'instrumentalness','liveness','speechiness','duration_ms']

    influencer=artist_data.loc[artist_data['artist_name']==name,:][data]
    X=influenced[data]

    X=pd.concat([influencer,X])
    X = (X-X.min())/(X.max()-X.min())

    influencer=X.loc[influencer.index,:]
    influencer=pd.DataFrame(influencer.values.T, index=influencer.columns, columns=influencer.index)
    column=influencer.columns[0]
    influencer=influencer[column]

    X=X.drop(index=column)
    change=(X-influencer)*(X-influencer)
    
    plt.figure(figsize=(20, 8))
    plt.plot(influencer,label='influencer')
    plt.plot(X.mean(),label='follower')
    plt.plot(change.mean(),label='change')
    plt.legend(prop={'size':24})
    
    return change.mean()


def influence_score_id(id_):
    followers=np.array(influence_data.loc[influence_data['influencer_id']==id_,:]['follower_id'])

    influenced = pd.DataFrame(columns=artist_data.columns)

    for follower in followers:    
        influenced=pd.concat([influenced,artist_data.loc[artist_data['artist_id'] == follower, :]])
    
    data=['danceability','energy','valence','tempo','loudness','mode','key','acousticness',
          'instrumentalness','liveness','speechiness','duration_ms']

    influencer=artist_data.loc[artist_data['artist_id']==id_,:][data]
    X=influenced[data]

    X=pd.concat([influencer,X])
    X = (X-X.min())/(X.max()-X.min())

    influencer=X.loc[influencer.index,:]
    influencer=pd.DataFrame(influencer.values.T, index=influencer.columns, columns=influencer.index)
    column=influencer.columns[0]
    influencer=influencer[column]

    X=X.drop(index=column)
    change=(X-influencer)*(X-influencer)
    
    plt.figure(figsize=(20, 8))
    plt.plot(influencer,label='influencer')
    plt.plot(X.mean(),label='follower')
    plt.plot(change.mean(),label='change')
    plt.legend(prop={'size':24})
    
    return change.mean()

'''
change=[]
change.append(influence_score_name('Bob Dylan'))
change.append(influence_score_name('The Beatles'))
change.append(influence_score_name('The Rolling Stones'))
change.append(influence_score_name('David Bowie'))
change.append(influence_score_name('Jimi Hendrix'))
'''
change=[]
change.append(influence_score_name('Big Star'))
change.append(influence_score_name('Little Richard'))
change.append(influence_score_name('George Jones'))
change.append(influence_score_name('Otis Redding'))
change.append(influence_score_name('Smokey Robinson'))

'''
PR_data=pd.read_csv('mapreduce法最后PR值.csv').sort_values(by='PR',ascending=False)
id_=np.array(PR_data['id'])
change_score=[]
for i in id_:
    change_score.append(influence_score())'''
    

'''
followers=np.array(influence_data.loc[influence_data['influencer_name']=='Bob Dylan',:]['follower_name'])

influenced = pd.DataFrame(columns=artist_data.columns)

for follower in followers:    
    influenced=pd.concat([influenced,artist_data.loc[artist_data['artist_name'] == follower, :]])
    
data=['danceability','energy','valence','tempo','loudness','mode','key','acousticness',
      'instrumentalness','liveness','speechiness','duration_ms']

influencer=artist_data.loc[artist_data['artist_name']=='Bob Dylan',:][data]
X=influenced[data]

X=pd.concat([influencer,X])
X = (X-X.min())/(X.max()-X.min())

influencer=X.loc[4,:]

X=X.drop(index=4)
loss=(X-influencer)*(X-influencer)


plt.figure(figsize=(20, 8))
plt.plot(influencer,label='influencer')
plt.plot(X.mean(),label='follower')
plt.plot(loss.mean(),label='loss')
plt.legend(prop={'size':24})
'''

