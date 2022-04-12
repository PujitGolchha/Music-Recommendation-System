import spotipy
import pickle
import csv
sp=spotipy.Spotify()
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Initialize API
client_credentials_manager = SpotifyClientCredentials(client_id='a987f9f942114637973f5992fd87768d',
                                                      client_secret="45225140405b42a1a259b89e3ea80f54")
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Generate list of songs to query metadata for
track_uri_toquery = pd.read_csv("C:/Users/Welcome/Desktop/college/sem6/project/spotify_playlist.csv")
list_to_query = list(track_uri_toquery['trackid1'])
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
              
    if (len(itemlist) == 20000): # save partial data into 600 chunks in case of error
        temp_filename = "audio_features" + str(i) + ".csv"
        print(temp_filename)
        my_df=pd.DataFrame(itemlist) 
        my_df.to_csv(temp_filename,index=False, header=True)
        del itemlist

