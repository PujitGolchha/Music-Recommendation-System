# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:10:24 2020

@author: Welcome
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:38:29 2020

@author: Welcome
"""
from tkinter import *
import tkinter as tk 
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.preprocessing import MinMaxScaler
from array import*
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict

dataset = pd.read_csv("Downloads/unique_songs_ty_1.csv")
dataset.head()

class word2vec():
    def __init__(self):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
    def generate_training_data(self, settings, corpus):
        training_data=[]
        word_counts = defaultdict(int)
    
        for row in corpus:
            for word in row:
                word_counts[word] += 1
        self.v_count = len(word_counts.keys())
        self.words_list = list(word_counts.keys())
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))	
        for sentence in corpus:
            sent_len= len(sentence)
            for i, word in enumerate(sentence):
                w_target = self.word2onehot(sentence[i])
                w_context = []
                for j in range(i - self.window, i + self.window+1):
                    if j != i and j <= sent_len-1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)
    def word2onehot(self, word):
		# word_vec - initialise a blank vector
        word_vec = [0 for i in range(0, self.v_count)] # Alternative - np.zeros(self.v_count)
		#############################
		# print(word_vec)			#
		# [0, 0, 0, 0, 0, 0, 0, 0]	#
		#############################

		# Get ID of word from word_index
        word_index = self.word_index[word]

		# Change value from 0 to 1 according to ID of the word
        word_vec[word_index] = 1
        return word_vec
    def train(self, training_data):
		# Initialising weight matrices
		# np.random.uniform(HIGH, LOW, OUTPUT_SHAPE)
		# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.uniform.html
		#self.w1 = np.array(getW1)
		#self.w2 = np.array(getW2)
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
		
		# Cycle through each epoch
        for i in range(self.epochs):
			# Intialise loss to 0
            self.loss = 0
			# Cycle through each training sample
			# w_t = vector for target word, w_c = vectors for context words
            for w_t, w_c in training_data:
				# Forward pass
				# 1. predicted y using softmax (y_pred) 2. matrix of hidden layer (h) 3. output layer before softmax (u)
                y_pred, h, u = self.forward_pass(w_t)
				#########################################
				# print("Vector for target word:", w_t)	#
				# print("W1-before backprop", self.w1)	#
				# print("W2-before backprop", self.w2)	#
				#########################################

				# Calculate error
				# 1. For a target word, calculate difference between y_pred and each of the context words
				# 2. Sum up the differences using np.sum to give us the error for this particular target word
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
				#########################
				# print("Error", EI)	#
				#########################

				# Backpropagation
				# We use SGD to backpropagate errors - calculate loss on the output layer 
                self.backprop(EI, h, w_t)
				#########################################
				#print("W1-after backprop", self.w1)	#
				#print("W2-after backprop", self.w2)	#
				#########################################

				# Calculate loss
				# There are 2 parts to the loss function
				# Part 1: -ve sum of all the output +
				# Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
				# Note: word.index(1) returns the index in the context word vector with value 1
				# Note: u[word.index(1)] returns the value of the output layer before softmax
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
				
				#############################################################
				# Break if you want to see weights after first target word 	#
				# break 													#
				#############################################################
            print('Epoch:', i, "Loss:", self.loss)
    def forward_pass(self, x):
		# x is one-hot vector for target word, shape - 9x1
		# Run through first matrix (w1) to get hidden layer - 10x9 dot 9x1 gives us 10x1
        h = np.dot(x, self.w1)
		# Dot product hidden layer with second matrix (w2) - 9x10 dot 10x1 gives us 9x1
        u = np.dot(h, self.w2)
		# Run 1x9 through softmax to force each element to range of [0, 1] - 1x8
        y_c = self.softmax(u)
        return y_c, h, u
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backprop(self, e, h, x):
		# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.outer.html
		# Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
		# Going backwards, we need to take derivative of E with respect of w2
		# h - shape 10x1, e - shape 9x1, dl_dw2 - shape 10x9
		# x - shape 9x1, w2 - 10x9, e.T - 9x1
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
		########################################
		# print('Delta for w2', dl_dw2)			#
		# print('Hidden layer', h)				#
		# print('np.dot', np.dot(self.w2, e.T))	#
		# print('Delta for w1', dl_dw1)			#
		#########################################

		# Update weights
        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)

	# Get vector from word
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

	# Input vector, returns nearest word(s)
    def vec_sim(self, word, top_n):
        v_w1 = self.word_vec(word)
        word_sim = {}
        for i in range(self.v_count):
			# Find the similary score for each word in vocab
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den
            word = self.index_word[i]
            word_sim[word] = theta
        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
        final= []
        for word, sim in words_sorted[:top_n]:
            wor=word
            final.append(wor)
            print(word, sim)
        return final
    
def prep(trackname):
    trackname=trackname.lower()
    trackname=trackname.split("(",1)[0]
    trackname=trackname.strip()
    return trackname

def playlist_format(playlists):
    documents=[]
    for i in range(0,max(playlists["playlistnumber"])+1):
        x=[]
        dataset=playlists.loc[playlists["playlistnumber"]==i]
        for i in range(0,len(dataset)):
            preprocessed=dataset.iloc[i,16]
            x.append(preprocessed)
        documents.append(x)
    return documents

    
def myfunction(canvas):
    canvas.configure(scrollregion=canvas.bbox("all"))   
def create_window():
    global x
    list=window.pack_slaves()
    for l in list:
        l.destroy() 
#    label = tk.Label(window, text="Select the songs", font=("Arial",30),background="grey10",foreground="lime green").pack(fill=X)
    canvas=tk.Canvas(window,background="grey10")
    frame=tk.Frame(canvas,background="grey10")
    myscrollbar=tk.Scrollbar(window,orient="horizontal",command=canvas.xview)
    canvas.configure(xscrollcommand=myscrollbar.set)  
    myscrollbar.pack(side="bottom",fill="x")
    canvas.pack(side="top",fill="both",expand=True)
    canvas.create_window((4,4),window=frame,anchor='nw')
    frame.bind("<Configure>",lambda event ,canvas=canvas:myfunction(canvas))
    k=0
    j=1
    for i in range(0,len(x)):
        if i%20==0:
            j=j+1
            k=0
        var=tk.IntVar()
        z.append(tk.Checkbutton(frame,text=x.iloc[i],font=("Arial"),background="grey10",foreground="lime green",variable=var).grid(row=j,column=k,sticky=W))
        z[-1]=var
        k=k+1
    tk.Button(frame,text="Next",font=("Arial"),background="lime green",foreground="grey10",width=30,command=lambda:[get_value(canvas)]).grid(row=j+1,column=0,sticky=W)
    
def get_value(canvas):
    selection=[]
    global newplaylist,dataset1,dataset2,dataset3,subset1000,new
    global co_mat,co_mat_sparse,col_filter,co_mat_transpose
    for i,c in enumerate(z):
        print(c.get())
        if c.get()==1:
            selection.append(dataset.loc[dataset["songname"]==x[i]])
    newplaylist=pd.concat(selection)
    newplaylist.iloc[:,1]=playlistno
    newplaylist=newplaylist.iloc[:,1:17]
    play=newplaylist[["songname","artistname"]]
    dataset1=dataset1.append(newplaylist)
    dataset1=dataset1.reset_index(drop=True)
    dataset2=dataset1.copy()
    subset1000=dataset1.copy()
    dataset3=dataset1.copy()
    dataset3["songname"]=dataset3["songname"].copy().apply(prep)
#    dataset3["artistname"]=dataset3["artistname"].map(lambda x:x.lower())
    dataset3["songartist"]=dataset3["songname"]+"-"+dataset3["artistname"]
    new=playlist_format(dataset3)
    co_mat = pd.crosstab(subset1000.playlistnumber,subset1000.id)
    co_mat = co_mat.clip(upper=1)
    assert np.max(co_mat.describe().loc['max']) == 1
    co_mat_sparse = csr_matrix(co_mat)
    co_mat_transpose=co_mat.transpose()
    col_filter = NearestNeighbors(metric='cosine', algorithm='brute')
    canvas.destroy()
    label = tk.Label(window, text="Songs Selected", font=("Arial",30),background="grey10",foreground="lime green").pack(fill=X)
    # create Treeview with 3 columns
    style=ttk.Style()
    style.theme_use("clam")
    #style.configure(".",foreground="white")
    style.configure("x.Treeview",font=('Arial',10,"bold"),background="lime green",fieldbackground="lime green",foreground="grey10")
    style.configure("x.Treeview.Heading",font=('Arial',12,"bold"),foreground="grey10",background="lime green",relief="flat")
    cols =('Song', 'Artistname')
    listBox = ttk.Treeview(window, columns=cols,height=20,show='headings',style="x.Treeview")
    # set column headings
    for col in cols:
        listBox.heading(col, text=col)    
    #listBox.grid(row=1, column=0, columnspan=2)
    listBox.pack(fill=BOTH)
    for i in range(0,len(play)):
        listBox.insert("", "end", values=(play.iloc[i,0], play.iloc[i,1]))
    #closeButton = tk.Button(window, text="Close", width=15, command=exit).grid(row=4, column=1)
    listBox.column("Song",anchor=CENTER)
    listBox.column("Artistname",anchor=CENTER)
    tk.Checkbutton(window,text="NLP",font=('Arial',12),variable=nlp,background="grey10",foreground="lime green").pack()
    tk.Checkbutton(window,text="Clustering",font=('Arial',12),variable=cluster,background="grey10",foreground="lime green").pack()
    tk.Checkbutton(window,text="Content Filtering",font=('Arial',12),variable=content,background="grey10",foreground="lime green").pack() 
    tk.Checkbutton(window,text="Collabarative Filtering",font=('Arial',12),variable=colab,background="grey10",foreground="lime green").pack()
    tk.Button(window,text="Next",font=('Arial',12),background="lime green",foreground="grey10",width=30,command=lambda:[filter1()]).pack()
    
def filter1():
    global dataset1,dataset2,dataset3
    if content.get()==1:
        val1=contentfiltering(dataset1)
#        window1=tk.Tk()
        title="Content Filtering Recommendations"
#        txt=tk.Text(window1)
#        txt.insert(tk.END,str(val1))
#        txt.pack()
        table(title,val1)
    if colab.get()==1:
        val4=useritemhybrid(playlistno)
        val3=usertouser(playlistno)
#        window2=tk.Tk()
        title="CollabarativeFiltering Recommendations:User-User"
#        txt=tk.Text(window2)
#        txt.insert(tk.END,str(val3))
#        txt.pack()
        table(title,val3)
#        window4=tk.Tk()
        title1="CollabarativeFiltering Recommendations:User-Item"
#        txt=tk.Text(window4)
#        txt.insert(tk.END,str(val4))
#        txt.pack()
        table(title1,val4)
    if cluster.get()==1:
        val2=clustering(dataset2)
#        window3=tk.Tk()
        title="Clustering Recommendations"
#        txt=tk.Text(window3)
#        txt.insert(tk.END,str(val2))
#        txt.pack()
        table(title,val2)
    if nlp.get()==1:
        val5=neural()
#        window5=tk.Tk()
        title="NLP Recommendations"
#        txt=tk.Text(window5)
#        txt.insert(tk.END,str(val5))
#        txt.pack()
        table(title,val5)

def table(heading,play):
    window1=tk.Tk()
    window1.title(heading)
    window1.geometry("700x400")
    label = tk.Label(window1, text=heading, font=("Arial",30),background="grey10",foreground="lime green").pack(fill=X)
    # create Treeview with 3 columns
    style1=ttk.Style()
#    style1.theme_use("clam")
    #style.configure(".",foreground="white")
    style1.configure("s.Treeview",font=('Arial',10,"bold"),background="lime green",fieldbackground="lime green",foreground="grey10")
    style1.configure("s.Treeview.Heading",font=('Arial',12,"bold"),foreground="grey10",background="lime green",relief="flat")
    cols =('Song', 'Artistname')
    listBox = ttk.Treeview(window1, columns=cols,height=20,show='headings',style="s.Treeview")
    # set column headings
    for col in cols:
        listBox.heading(col, text=col)    
    #listBox.grid(row=1, column=0, columnspan=2)
    listBox.pack(fill=BOTH)
    for i in range(0,len(play)):
        listBox.insert("", "end", values=(play.iloc[i,0], play.iloc[i,1]))
    #closeButton = tk.Button(window, text="Close", width=15, command=exit).grid(row=4, column=1)
    listBox.column("Song",anchor=CENTER)
    listBox.column("Artistname",anchor=CENTER)

def contentfiltering(x):
    global playlistno
    subset100=x
    features_drop = ["songname","artistname","id","playlistnumber"]
    train_cleaned = subset100.drop(features_drop, axis =1)#dropping unimportant columns i.e axis=1
    train_cleaned.head()
    scaler = MinMaxScaler()
    scaler.fit(train_cleaned)
    train_scaled = scaler.transform(train_cleaned)#standardizing the data
    cos=np.zeros((250,len(train_scaled)))#creating an empty matrix 
    j=0
    playlistnumber=playlistno#the target playlist number
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
    
    index=index%len(train_scaled)#dividing the index by no of rows to get remainder, the remainder is the real index of the songs 
    
    unique_candidate_song_sorted =subset100['id'][index][::-1].drop_duplicates()#get the unique songs in descending order
    
    tracks_in_target_playlist = subset100.loc[subset100["playlistnumber"] ==playlistnumber, "id"]#get the tracks in target playlist
    
    song_to_recommend = np.array(unique_candidate_song_sorted.loc[~unique_candidate_song_sorted.isin(tracks_in_target_playlist)])#get the songs that are not in target paylist in descending order
    
    song_to_recommend = song_to_recommend[:10]#top 10 songs to recommend
    
    for i in range(0,len(song_to_recommend)):##loop to create a dataframe of all the recommended songs for the given playlist
        if i==0:
            final_recommendation1=subset100.loc[subset100["id"] == song_to_recommend[i]]
            final_recommendation1=pd.DataFrame((final_recommendation1.iloc[0]))
            final_recommendation1=final_recommendation1.T
        else:
            values1=subset100.loc[subset100["id"] == song_to_recommend[i]]
            values1=pd.DataFrame((values1.iloc[0]))
            values1=values1.T
            final_recommendation1=final_recommendation1.append(values1)
    final_recommendation1=final_recommendation1[["songname","artistname"]]
    return final_recommendation1

def clustering(x):
    final=x
    global playlistno
#    final['index'] = np.arange(1, len(final)+1) 
#    scaleCols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
#                 'key', 'liveness', 'loudness', 'speechiness', 'tempo','valence','mode',"duration_ms"] #mode excluded from analysis
#    scaler = StandardScaler()
#    scaler.fit(final.loc[:, scaleCols])
#    train_scaled = final.copy() #copy original master data frame
#    train_scaled[scaleCols] = scaler.transform(train_scaled[scaleCols])#scale transform cluster columns
#    train_scaled['index'] = np.arange(1, len(train_scaled)+1) #reappend index column
#    train_scaled = train_scaled.rename(columns = {'acousticness': 'acousticness_scaled',
#                                                  'danceability': 'danceability_scaled',
#                                                  'energy': 'energy_scaled',
#                                                  'instrumentalness': 'instrumentalness_scaled',
#                                                  'key': 'key_scaled',
#                                                  'liveness': 'liveness_scaled',
#                                                  'loudness': 'loudness_scaled',
#                                                  'speechiness': 'speechiness_scaled',
#                                                  'tempo': 'tempo_scaled',
#                                                  'valence': 'valence_scaled',
#                                                  'mode': 'mode_scaled',
#                                                  'duration_ms':'duration_ms_scaled'})
#    joinCols =["index","playlistnumber","artistname","songname","id"]
#    final_new = final.merge(train_scaled, on = joinCols, how = 'outer')
#    final1=final_new.copy()
#    final1=final1['id'].drop_duplicates()
#    values=final_new.iloc[final1.index]
#    values.to_csv("C:/Users/Welcome/Desktop/college/sem6/project/unique_songs_ty.csv",header=True)
    #
    values_new=pd.read_csv("Downloads/unique_songs_ty_1.csv")
    clusterCols = ['acousticness_scaled','danceability_scaled', 
                   'energy_scaled', 'instrumentalness_scaled',
                   'key_scaled', 'liveness_scaled', 'loudness_scaled',
                   'speechiness_scaled', 'tempo_scaled',
                   'valence_scaled', 'mode_scaled','duration_ms_scaled'] #variables to cluster
    
    kmeans = KMeans(n_clusters = 5,random_state=5)
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
    
    final2 = final.merge(values_new[['id','cluster_label']], on = ['id'])
    final2.sort_values(by=['playlistnumber'])
    pn=playlistno
    subset=final2[final2.playlistnumber==pn]
    clusterlabel = subset.groupby(['cluster_label'])['playlistnumber'].count().reset_index().sort_values('playlistnumber').tail(1).iloc[0,0]
    count=0
    artists=subset.groupby(['artistname'])['playlistnumber'].count().reset_index().sort_values('playlistnumber')
    artistlabel = defaultdict(list)
    for i in range(len(artists)-1,-1,-1):
            artist=artists.iloc[i,0]
            count=count+artists.iloc[i,1]
            artistlabel['names'].append(artist)  
            
    for i in range(0,len(artistlabel['names'])):
        y=final2[(final2['cluster_label']==clusterlabel) & (final2['artistname']==artistlabel['names'][i])]
        if i==0:
            recommendation=y
        else:
            recommendation=recommendation.append(y)
          
    recommendation=recommendation['id'].drop_duplicates()
    tracks_in_target_playlist = final2.loc[final2["playlistnumber"] == pn, "id"]
    song_to_recommend = recommendation.loc[~recommendation.isin(tracks_in_target_playlist)]
    final_recommendation2=final2.iloc[song_to_recommend.index]
    
    if len(final_recommendation2)>10:
        final_recommendation2=final_recommendation2.sample(n=10)
    
    final_recommendation2=final_recommendation2[["songname","artistname"]]
    return final_recommendation2

def usertouser(playlist_id):
    global subset1000,co_mat,co_mat_sparse,col_filter
    knnmodel=col_filter.fit(co_mat_sparse)
    k = 10
    ref_songs = co_mat.columns.values[co_mat.loc[playlist_id] == 1] # songs already in playlist
    dist, ind = knnmodel.kneighbors(np.array(co_mat.loc[playlist_id]).reshape(1, -1), n_neighbors = 10)
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
    pred=pd.DataFrame(pred,columns={"id"})
    all1=pred.join(subset1000.set_index('id'),on = 'id')
    recommendation=all1.drop_duplicates('id')[['songname','artistname']]
    return recommendation

def useritemhybrid(playlist_id):
    global subset1000,co_mat,co_mat_transpose,col_filter
    knnmodel=col_filter.fit(co_mat_transpose)
    songs= co_mat.columns.values[co_mat.loc[playlist_id] == 1]
    rec_ind=pd.DataFrame()
    for i in songs:
        dist, ind = knnmodel.kneighbors(np.array(co_mat_transpose.loc[i]).reshape(1, -1), n_neighbors = 10)
        rec_songs= co_mat_transpose.index[ind[0]] # recommended songs
        rec_ind=rec_ind.append(pd.DataFrame(data=rec_songs))
    
    preds = pd.DataFrame(np.reshape(rec_ind, (len(rec_ind),1)))
    rec=preds.id.value_counts()
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
    pred=pd.DataFrame(pred,columns={"id"})
    all1=pred.join(subset1000.set_index('id'),on = 'id')
    recommendation=all1.drop_duplicates('id')[['songname','artistname']]
    return recommendation

def neural():
    global settings, new, dataset3,playlistno
    training_data = w2v.generate_training_data(settings, new)
    print(training_data)
    # Training
    w2v.train(training_data)
    # Get vector for word
    tracks_in_target_playlist = dataset3.loc[dataset3["playlistnumber"] ==playlistno, "songartist"]
    z=[]
    for word in tracks_in_target_playlist:
         z.append(w2v.vec_sim(word, 2))
    
    recommend=np.array(z)
    recommend=pd.DataFrame(recommend)
    recommendation=recommend.iloc[:,1]
    recommendation= recommendation.loc[~recommendation.isin(tracks_in_target_playlist)]
    song_to_recommend = dataset3.loc[dataset3["songartist"].isin(recommendation)]     
    song_to_recommend= song_to_recommend.drop_duplicates(subset=['songname'])
    final_recommendation=song_to_recommend[["songname","artistname"]]
    return final_recommendation
    
z=[]
x=dataset.iloc[:,2]
newplaylist=[]
dataset1 = pd.read_csv("Downloads/ty's_playlists.csv")
dataset2=pd.DataFrame()
dataset3=pd.DataFrame()
subset1000=pd.DataFrame()
settings = {
	'window_size': 2,			# context window +- center word
	'n': 30,					# dimensions of word embeddings, also refer to size of hidden layer
	'epochs': 25,				# number of training epochs
	'learning_rate': 0.01		# learning rate
}
w2v = word2vec()
playlistno=dataset1["playlistnumber"].max()+1
new=[]
co_mat=[]
co_mat_sparse=[]
col_filter=[]
co_mat_transpose=[]
window=tk.Tk() 
window.title("Recommender System")
window.configure(background="grey10")
content=tk.IntVar()
colab=tk.IntVar()
cluster=tk.IntVar()
nlp=tk.IntVar()
window.geometry("2000x2000")
zero=tk.Button(window,text="Make a playlist",font=("Arial",30),background="grey10",foreground="lime green",height=20,command=lambda:[create_window()]).pack(fill=BOTH)
window.mainloop()