import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
pd.options.display.max_columns=None
import os
final = pd.read_excel("C:/Users/Welcome/Desktop/college/sem6/project/spotifyplaylist+features.xlsx")
#
features_drop = ["duration_ms","playlists__duration_ms","track_id"]
final=final.drop(features_drop,axis=1)
final['index'] = np.arange(1, len(final)+1) 
scaleCols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
             'key', 'liveness', 'loudness', 'speechiness', 'tempo','valence','mode'] #mode excluded from analysis
scaler = StandardScaler()
scaler.fit(final.loc[:, scaleCols])
train_scaled = final.copy() #copy original master data frame
train_scaled[scaleCols] = scaler.transform(train_scaled[scaleCols])#scale transform cluster columns
train_scaled['index'] = np.arange(1, len(train_scaled)+1) #reappend index column
train_scaled = train_scaled.rename(columns = {'acousticness': 'acousticness_scaled',
                                              'danceability': 'danceability_scaled',
                                              'energy': 'energy_scaled',
                                              'instrumentalness': 'instrumentalness_scaled',
                                              'key': 'key_scaled',
                                              'liveness': 'liveness_scaled',
                                              'loudness': 'loudness_scaled',
                                              'speechiness': 'speechiness_scaled',
                                              'tempo': 'tempo_scaled',
                                              'valence': 'valence_scaled',
                                              'mode': 'mode_scaled'})
joinCols =["index","playlists__pid","playlists__tracks__artist_name","playlists__tracks__track_name","playlists__tracks__album_name","playlists__tracks__duration_ms","id"]
final_new = final.merge(train_scaled, on = joinCols, how = 'outer')
final1=final_new.copy()
final1=final1['id'].drop_duplicates()
values=final_new.iloc[final1.index]
values.to_csv("C:/Users/Welcome/Desktop/college/sem6/project/unique_songs.csv",header=True)
#
values_new=pd.read_csv("C:/Users/Welcome/Desktop/college/sem6/project/unique_songs.csv")
clusterCols = ['acousticness_scaled','danceability_scaled', 
               'energy_scaled', 'instrumentalness_scaled',
               'key_scaled', 'liveness_scaled', 'loudness_scaled',
               'speechiness_scaled', 'tempo_scaled',
               'valence_scaled', 'mode_scaled'] #variables to cluster

kmeans = KMeans(n_clusters = 5)
kmeans.fit(values_new.loc[:, clusterCols])
center = kmeans.cluster_centers_
label = kmeans.labels_
values_new['cluster_label'] = label
values_new['cluster_label'] = values_new['cluster_label'] + 1
centroids = defaultdict(list)
for col in clusterCols:
    centroids['columns'].append(col)
for a in range(len(center)):
    for b in range(len(center[0])):
        centroids['c'+ str(a)].append(center[a][b])
x=pd.DataFrame(centroids)

prediction_cluster = values_new[['playlists__pid','playlists__tracks__artist_name','playlists__tracks__track_name',
                                 'playlists__tracks__duration_ms','playlists__tracks__album_name','cluster_label','id']]

mode_artist = prediction_cluster.groupby(['cluster_label', 'playlists__tracks__artist_name'])['playlists__pid'].count().reset_index()
mode_artist = mode_artist.rename(columns = {'playlists__pid': 'mode_artist'})
prediction_cluster = prediction_cluster.merge(mode_artist, on = ['cluster_label', 'playlists__tracks__artist_name'])
final2 = final.merge(prediction_cluster[['id','cluster_label','mode_artist']], on = ['id'])
final2.sort_values(by=['playlists__pid'])
pn=2
subset=final2[final2.playlists__pid==pn]
clusterlabel = subset.groupby(['cluster_label'])['playlists__pid'].count().reset_index().sort_values('playlists__pid').tail(3).iloc[2,0]
count=0
artists=subset.groupby(['playlists__tracks__artist_name'])['playlists__pid'].count().reset_index().sort_values('playlists__pid').tail(5)
artistlabel = defaultdict(list)
for i in range(4,-1,-1):
        artist=artists.iloc[i,0]
        count=count+artists.iloc[i,1]
        artistlabel['names'].append(artist)  
        
for i in range(0,len(artistlabel['names'])):
    y=final2[(final2['cluster_label']==clusterlabel) & (final2['playlists__tracks__artist_name']==artistlabel['names'][i])]
    if i==0:
        recommendation=y
    else:
        recommendation=recommendation.append(y)
      
recommendation=recommendation['id'].drop_duplicates()
tracks_in_target_playlist = final2.loc[final2["playlists__pid"] == pn, "id"]
song_to_recommend = recommendation.loc[~recommendation.isin(tracks_in_target_playlist)]
final_recommendation=final2.iloc[song_to_recommend.index]
final_recommendation=final_recommendation.sample(n=10)
