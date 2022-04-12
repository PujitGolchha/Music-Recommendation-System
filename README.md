# Music-Recommendation-System

The Project's aim is to build a music recommendation system using multiple different algorithms.

Data: 4000 user playlists from Spotify through their RecSys challenge. Additional features for each song scraped using Spotify's API. 

#####################################################################

Algorithms used:

Content Filtering: Closest songs to the songs in a playlist are found based on their extracted features. The top k songs which are not already part of the playlist are the new recommendations.

User to User Collabarative Filtering: Using the bag of words approach, a vector representation of each playlist is created. Each position in the vector represnts a song and the value of each positon depends on if the song is present in the playlist or not (0's for the songs not in plyalist and 1's for the song in the playlist). The K-nearest neighbours of the target playlist are found using Euclidean distance. Songs from those playlists which are not already part of the playlist are the new recommendations.

User to Item Collabarative Filtering:  Using the bag of words approach, a vector representation of each unique song is created. Each position in the vector represents a playlist and the value of each positon depends on if the song is present in the playlist or not (0's for the songs not in plyalist and 1's for the song in the playlist). The K-nearest neighbours for each song in the playlist were found using Euclidean distance. The songs that occur most frequently as nearest neighbours and  which are not already part of the playlist are the new recommendations.

Clustering: All the songs are clustered into 5 different clusters. For each playlist the cluster which has most number of songs from the playlist is found. Songs from that cluster of those artists which are already in the playlist and which are not already part of the playlist are the new recommendations.

Word2Vec model: For each song we get a vector represntation using the skipgram model i.e by trying to train a neural network to predict two songs which are played before and after the particular song. (An assumption made here was that the songs in the playlist are in the order in which someone listens to them). The K-nearest neighbours for each song in the playlist were found using Euclidean distance. Songs from those artists which are already in the playlist and which are not already part of the playlist are the new recommendations.


#####################################################################
 
A GUI which uses tkinter is also avalaible

A person just needs to have the datasets "playlists.csv" which are 25 hindi playlists and  "unique_songs.csv" which are the unique songs from these playlists. 
Firstly the paths to these files need updated in the path1 and path2 variables in the GUI.py and then the GUI.py can be run.
In real time the user can choose songs which he likes and based on those songs he will get recommendations using different algorithms from the unique songs list.



#####################################################################

