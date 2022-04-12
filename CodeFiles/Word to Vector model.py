

@author: Welcome
"""
import numpy as np
import pandas as pd
from collections import defaultdict


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

        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec
    
    def train(self, training_data):
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
        for i in range(self.epochs):
			# Intialise loss to 0
            self.loss = 0
            for w_t, w_c in training_data:
				y_pred, h, u = self.forward_pass(w_t)
				EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
				self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
            print('Epoch:', i, "Loss:", self.loss)
    
    def forward_pass(self, x):
        h = np.dot(x, self.w1)
        u = np.dot(h, self.w2)
        y_c = self.softmax(u)
        return y_c, h, u
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backprop(self, e, h, x):
		dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
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

#####################################################################
settings = {
	'window_size': 2,			# context window +- center word
	'n': 30,					# dimensions of word embeddings, also refer to size of hidden layer
	'epochs': 100,				# number of training epochs
	'learning_rate': 0.01		# learning rate
}

data=pd.read_csv("C:/Users/Ananya/Documents/NMIMS/Projects Sem 6/Data Science/Spotify dataset/spotify playlist and features.xlsx")

def prep(trackname):
    trackname=trackname.lower()
    trackname=trackname.split("(",1)[0]
    trackname=trackname.strip()
    return trackname

data["songname"]=data["playlists__tracks__track_name"].copy().apply(prep)
data["artistname"]=data["playlists__tracks__artist_name"].map(lambda x:x.lower())
data["songartist"]=data["songname"]+"-"+data["artistname"]

def playlist_format(playlists):
    documents=[]
    for index,row in playlists.iterrows():
        preprocessed=row["songartist"]
        documents.append(preprocessed)
    return documents

new=playlist_format(data)

# Initialise object
w2v = word2vec()

# Numpy ndarray with one-hot representation for [target_word, context_words]
training_data = w2v.generate_training_data(settings, new)
print(training_data)
# Training
w2v.train(training_data)
# Get vector for word
playlistnumber=13
tracks_in_target_playlist = data.loc[data["playlistnumber"] ==playlistnumber, "songartist"]
z=[]
for word in tracks_in_target_playlist:
     z.append(w2v.vec_sim(word, 2))

recommend=np.array(z)
recommend=pd.DataFrame(recommend)
recommendation=recommend.iloc[:,1]
recommendation= recommendation.loc[~recommendation.isin(tracks_in_target_playlist)]
song_to_recommend = data.loc[data["songartist"].isin(recommendation)]     
song_to_recommend= song_to_recommend.drop_duplicates(subset=['songname'])
final_recommendation=song_to_recommend[["songname","artistname"]]