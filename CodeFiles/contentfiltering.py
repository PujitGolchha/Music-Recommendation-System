# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 19:03:55 2020

@author: Welcome
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.preprocessing import MinMaxScaler
from array import*
from scipy.sparse import csr_matrix
from tqdm import tqdm
subset100 = pd.read_excel("C:/Users/Welcome/Desktop/college/sem6/project/spotifyplaylist+features.xlsx")
subset100.head()
features_drop = ["playlists__pid","playlists__tracks__artist_name","playlists__tracks__track_name","playlists__tracks__album_name","playlists__tracks__duration_ms","playlists__duration_ms","id","track_id"]
train_cleaned = subset100.drop(features_drop, axis =1)#dropping unimportant columns i.e axis=1
train_cleaned.head()
scaler = MinMaxScaler()
scaler.fit(train_cleaned)
train_scaled = scaler.transform(train_cleaned)#standardizing the data
cos=np.zeros((250,266386))#creating an empty matrix 
j=0
playlistnumber=41#the target playlist number
for i in tqdm(range(0,len(train_scaled))):#this loop is used to find the cosine similarity matrix only for the songs in the inputed playlist number
    if(subset100.iloc[i,0]==playlistnumber):#here the playlist number is checked
        y=train_scaled[i,].reshape(1,12)#reshaping the row to y 
        x=train_scaled[0:len(train_scaled),]#x is the matrix of all songs
        train_scaled_cos_matrix = cosine_similarity(x,y)
        m=np.transpose(train_scaled_cos_matrix)
        cos[j,]=m#storing each row into the cosine similarity matix
        j=j+1 
        #the first n rows of the cos matrix are the cosine similarities between songs of the given playlist and the entire data where n is the size of the playlist
        #the other rows are 0
cos=np.array(cos)

cos1=np.array(cos).flatten()#flatten the cosine matrix

u,index=np.unique(cos1,return_index=True)#get the unique values in ascending order and their index values

index=index%266386#dividing the index by no of rows to get remainder, the remainder is the real index of the songs 

unique_candidate_song_sorted =subset100['id'][index][::-1].drop_duplicates()#get the unique songs in descending order

tracks_in_target_playlist = subset100.loc[subset100["playlists__pid"] ==playlistnumber, "id"]#get the tracks in target playlist

song_to_recommend = np.array(unique_candidate_song_sorted.loc[~unique_candidate_song_sorted.isin(tracks_in_target_playlist)])#get the songs that are not in target paylist in descending order

song_to_recommend = song_to_recommend[:10]#top 10 songs to recommend

for i in range(0,len(song_to_recommend)):##loop to create a dataframe of all the recommended songs for the given playlist
    if i==0:
        values=subset100.loc[subset100["id"] == song_to_recommend[i],["playlists__tracks__track_name","playlists__tracks__artist_name","playlists__tracks__album_name","playlists__tracks__duration_ms"]]
        values=pd.DataFrame((values.iloc[0]))
        values=values.T
    else:
        values1=subset100.loc[subset100["id"] == song_to_recommend[i],["playlists__tracks__track_name","playlists__tracks__artist_name","playlists__tracks__album_name","playlists__tracks__duration_ms"]]
        values1=pd.DataFrame((values1.iloc[0]))
        values1=values1.T
        values=values.append(values1)
        
