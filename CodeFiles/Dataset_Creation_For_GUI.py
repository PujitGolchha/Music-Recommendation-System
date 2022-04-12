# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:33:12 2020

@author: Welcome
"""
import pandas as pd
import numpy as np
import sys
import spotipy
import spotipy.util as util
import csv
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyClientCredentials
sp=spotipy.Spotify()
username="9v77veob0pxqb8t7bfhzb5b89"
client_credentials_manager = SpotifyClientCredentials(client_id='29eeace961b147aca1751045687de7d0',
                                                      client_secret="ca48720bd082431684d9c2b4e50619a0")
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)              
playlists = sp.user_playlists(user=username)
for i,playlist in enumerate(playlists['items']):
    print()
    print(playlist['name'])
    print ('total tracks', playlist['tracks']['total'])
    y=np.array(playlist['id'])
    if i==0:
        recommendation=y
    else:
        recommendation=np.append(recommendation,y)
dataset = pd.read_csv("C:/Users/Welcome/Desktop/college/sem6/project/unique_songs_ty_1.csv")
songslist=pd.DataFrame(columns=['playlistnumber','songname','id'])
k=0
for i in range(0,25):
    x=sp.user_playlist_tracks(user=username,playlist_id=recommendation[i])
    print(len(x['items']))
    for j in range(0,len(x['items'])):
        songslist.loc[k,"playlistnumber"]=i
        songslist.loc[k,"songname"]=x['items'][j]['track']['name']
        songslist.loc[k,"id"]=x['items'][j]['track']['id']
        k=k+1
joinCols =["songname","id"]
songslist_new = songslist.merge(dataset[["songname","id","artistname"]], on = joinCols, how = 'left')
songslist_new.to_csv("C:/Users/Welcome/Desktop/college/sem6/project/ty's_playlists.csv",index=False, header=True)



track_uri_toquery = pd.read_csv("C:/Users/Welcome/Desktop/college/sem6/project/ty's_playlists.csv")
list_to_query = list(track_uri_toquery['id'])
unique_tracks = list(set(list_to_query))
start = 0
end = len(list_to_query)

        
for i in range(0,end, 50):
    temp_list = list_to_query[i:i+50]
    playlists = sp.audio_features(tracks=temp_list)
    print(i)
    if "itemlist" in locals():
        itemlist = itemlist + playlists # append new data
    else:
        itemlist = playlists
final_new=[]
my_df=pd.DataFrame(itemlist)
final_new = track_uri_toquery.merge(my_df,left_index=True,right_index=True)
features_drop = ["analysis_url","time_signature","track_href","type","uri","id_y"]
final_new = final_new.drop(features_drop, axis =1)
final_new=final_new.rename(columns={"id_x":"id"})
final_new.to_csv("C:/Users/Welcome/Desktop/college/sem6/project/ty's_playlists.csv",index=False, header=True)