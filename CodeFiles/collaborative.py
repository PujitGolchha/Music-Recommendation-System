#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


# In[2]:


subset1000 = pd.read_excel("C:/Users/Ananya/Documents/NMIMS/Projects Sem 6/Data Science/Spotify dataset/spotify playlist and features.xlsx")
subset1000.head()


# In[3]:


# Create Binary Sparse Matrix
co_mat = pd.crosstab(subset1000.pid, subset1000.trackid)
co_mat = co_mat.clip(upper=1)
assert np.max(co_mat.describe().loc['max']) == 1

co_mat_sparse = csr_matrix(co_mat)


# In[4]:


co_mat_transpose=co_mat.transpose()
co_mat_transpose


# In[5]:


col_filter = NearestNeighbors(metric='cosine', algorithm='brute')


# In[ ]:


col_filter.fit(co_mat_sparse)


# In[10]:


def kpredictuseruser(knnmodel, playlist_id):
    
    k = 10
    ref_songs = co_mat.columns.values[co_mat.loc[playlist_id] == 1] # songs already in playlist
    dist, ind = knnmodel.kneighbors(np.array(co_mat.loc[playlist_id]).reshape(1, -1), n_neighbors = 49)
    rec_ind = co_mat.index[ind[0]] # recommended playlists
    
    n_pred = 0
    pred = []
    for i in rec_ind:
        new_songs = co_mat.columns.values[co_mat.loc[i] == 1] # potential recommendations
        for song in new_songs:
            if song not in ref_songs: # only getting songs not already in target playlist
                pred.append(song)
                n_pred += 1
                if n_pred == k:
                    break
        if n_pred == k:
            break
    pred=pd.DataFrame(pred,columns={"trackid"})
    all=pred.join(subset1000.set_index('trackid'),on = 'trackid')
    recommendation=all.drop_duplicates('trackid')[['playlists__tracks__track_name','playlists__tracks__artist_name']]
    return recommendation


# In[11]:



pi = 3067 # target playlist index
kpreds = kpredictuseruser(col_filter, pi) # list of predictions


# In[12]:


kpreds # user to user predictions


# In[6]:


col_filter.fit(co_mat_transpose)


# In[7]:


def kpredictitemitem(knnmodel, track_id):
    
    k = 10
    ref_playlists = co_mat_transpose.columns.values[co_mat_transpose.loc[track_id] == 1] # list of playlists where that song occurs
    dist, ind = knnmodel.kneighbors(np.array(co_mat_transpose.loc[track_id]).reshape(1, -1), n_neighbors = 49)
    rec_ind = co_mat_transpose.index[ind[0]] # recommended songs
    
    return rec_ind


# In[8]:


ti = '0UioblV1x795s55Ur58c6c' # target track
kpreds_playlist = kpredictitemitem(col_filter, ti)


# In[9]:


kpreds_playlist # trackids of item to item


# In[24]:


# User Item Hybrid
def kpredictuseritem(knnmodel, playlist_id):
    songs= co_mat.columns.values[co_mat.loc[playlist_id] == 1]
    rec_ind=pd.DataFrame()
    for i in songs:
        dist, ind = knnmodel.kneighbors(np.array(co_mat_transpose.loc[i]).reshape(1, -1), n_neighbors = 10)
        rec_songs= co_mat_transpose.index[ind[0]] # recommended songs
        rec_ind=rec_ind.append(pd.DataFrame(data=rec_songs))
    
    preds = pd.DataFrame(np.reshape(rec_ind, (len(rec_ind),1)))
    rec=preds.trackid.value_counts()
    final=pd.DataFrame(rec).reset_index()
    final=final.iloc[:,0]
    k=10
    n_pred = 0
    ref_songs = co_mat.columns.values[co_mat.loc[playlist_id] == 1] # songs already in playlist
    pred = []
    for song in final:
        if song not in ref_songs: # only getting songs not already in target playlist
            pred.append(song)
            n_pred += 1
            if n_pred == k:
                break
        if n_pred == k:
            break
    
    pred=pd.DataFrame(pred,columns={"trackid"})
    all=pred.join(subset1000.set_index('trackid'),on = 'trackid')
    recommendation=all.drop_duplicates('trackid')[['playlists__tracks__track_name','playlists__tracks__artist_name']]
    return recommendation


# In[25]:


finalpreds=kpredictuseritem(col_filter,3067) #user to item predictions


# In[23]:


finalpreds 

