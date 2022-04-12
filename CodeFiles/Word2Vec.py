# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:16:43 2020

@author: Ananya
"""
from gensim.models import Word2Vec

data=pd.read_excel("C:/Users/Ananya/Documents/NMIMS/Projects Sem 6/Data Science/Spotify dataset/spotify playlist and features.xlsx")
#data=data[data.pid<250]

def prep(trackname):
    trackname=trackname.lower()
    trackname=trackname.split("(",1)[0]
    trackname=trackname.strip()
    return trackname

data["songname"]=data["playlists__tracks__track_name"].copy().apply(prep)
#data["artistname"]=data["artistname"].map(lambda x:x.lower())
data["artistname"]=data["playlists__tracks__artist_name"]
data["songartist"]=data["songname"]+"-"+data["artistname"]

def playlist_format(playlists):
    documents=[]
    for i in range(0,max(playlists["pid"])+1):
        x=[]
        dataset=playlists.loc[playlists["pid"]==i]
        for i in range(0,len(dataset)):
            preprocessed=dataset.iloc[i,44]
            x.append(preprocessed)
        documents.append(x)
    return documents

new=playlist_format(data)

model = Word2Vec(new, min_count=1,size= 150,workers=3, window =2, sg = 1)
print(model)

words = list(model.wv.vocab)
print(words)

playlistnumber=13
tracks_in_target_playlist = data.loc[data["pid"] ==playlistnumber, "songartist"]
z=[]
for word in tracks_in_target_playlist:
     z.append(model.wv.similar_by_word(word, topn=2, restrict_vocab=None))

recommend=np.array(z).reshape((-1, 2))
recommend=pd.DataFrame(recommend)
recommendation=recommend.iloc[:,0]
recommendation= recommendation.loc[~recommendation.isin(tracks_in_target_playlist)]
song_to_recommend = data.loc[data["songartist"].isin(recommendation)]    
song_to_recommend= song_to_recommend.drop_duplicates(subset=['songname'])
final_recommendation=song_to_recommend[["songname","artistname"]]