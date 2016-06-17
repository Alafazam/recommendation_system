from scipy import spatial
from math import sqrt
from utils import *
from math import log10
import numpy as np
import os, datetime
import time,json
import random
import sqlite3

from similarityMeasures import cosine_similarity,pearson_similarity
from get_dataset import V, U, ALL_MOVIES, Movie_generes
from baseAlgos import greedy_filtering, nSquare_user_similarity_matrix
from baseAlgos import TF_IDF_normalize_ratings

db = 'MovieLens.db'
USE_DB =False
if USE_DB :
	if USE_DB and not os.path.isfile(db): crt = True
	else: crt = False
	conn = sqlite3.connect(db)
	c = conn.cursor()
	if crt:	c.execute('''CREATE TABLE greedyF_user_similarity_matrix(userID real, otherID real, similarity real)''')

def log(x):
	for q in range(len(x)):
			print x[q]
np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')

def displayUser(id,log=False):
	userID = id - 1
	mrated = V[userID]
	n = V.shape[1]
	_movies = [ ALL_MOVIES[movieId] for movieId in range(n) if V[userID][movieId]>0 ]
	# _names_of_rated = [x.getinfo() for x in _movies]
	user = U[userID]
	user_string = "User:   %s.\nAge  :%s.\nGender  :%s.\nOccupation  :%s.\nZipcode  %s"%(str(userID+1), str(user[1]), str(user[2]), str(user[3]), str(user[4]))
	if(log):
		print user_string
		print "Movies rated by selected user "
		print _movies[:10]
		print "    "
	return _movies, user_string

# displayUser(userid, True)


def find_k_similar_users(similarity_matrix, k = 20):
	similarUsers = [[] for q in xrange(len(similarity_matrix))]
	for x in xrange(len(similarity_matrix)):
		for y in xrange(len(similarity_matrix[x])):
			# print similarity_matrix[x][y]
			if similarity_matrix[x][y] > 0:
				similarUsers[x].append([similarity_matrix[x][y],y])
		similarUsers[x].sort()
		similarUsers[x] = similarUsers[x][::-1][:k]
	return similarUsers

def get_k_similar_users(similarity_matrix, userID, k = 20):
	similarUsers = []
	for x in similarity_matrix[userID]:
		if similarity_matrix[userID][x] > 0:
			similarUsers.append([similarity_matrix[userID][x],x])
	similarUsers.sort()
	similarUsers = similarUsers[::-1][:k]
	return similarUsers



def get_similar_users_from_DB(userid,k=50):
	c.execute("SELECT * FROM user_similarity_matrix WHERE userID=%d AND otherID!=%d ORDER BY similarity DESC LIMIT %d;"%(userid, userid, k))
	similarUsers = c.fetchall()
	similarUsers = np.array(similarUsers)
	similarUsers = np.delete(similarUsers, np.s_[0:1], 1)
	return similarUsers

# SU = get_similar_users_from_DB(552)
# print SU[40]

# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(V, userid, SU):
	V2 = V.reshape(1682,943)
	u_rated = V[userid].nonzero()[0]
	n = len(V[userid])
	n_SU = len(SU)
	# others_sim =  np.hsplit(SU,2)
	# others = others.reshape((n_SU)).astype(int)
	# others_sim = others_sim.reshape((n_SU))
	# print others_sim
	final_scores = []
	for x in range(n):
		if x in u_rated: continue
		score = 0
		o_n = 1
		# print x
		for othersID, others_sim in SU:
			# print othersID, others_sim
			others_rating = V2[x][int(othersID)]
			if others_rating > 0:
				score += (others_rating)*others_sim
				o_n += 1
		score = score/o_n
		final_scores.append([score, x])
	final_scores.sort()
	final_scores.reverse()
	final = [y for x, y in final_scores]
	return ALL_MOVIES[final][:10]



def RCF(matrix, userID, Q, k=30):
	ratings_by_user = np.array(matrix[userID])
	I_R = [ i for i in range(len(raings_by_user)) if raings_by_user[i] > 0]
	I_U = [ i for i in range(len(raings_by_user)) if raings_by_user[i] == 0]

	S = [ [] for i in range(len(I_U))]

	for i in range(len(ratings_by_user)):
		if i in I_R:
			for j, sim_j in Q[i]:
				S[j].append([sim_j,i])
			S[j].sort()
			S[j] = S[j][::-1]
		if i in I_U:
			if len(S[i]) > k:
				S[i] = S[i][:k]
	return S


for z in ALL_MOVIES:
	z.setRatings(V)

# movies by genres
Movie_by_generes = {}
for w in Movie_generes:
	Movie_by_generes[w] = []

for movie in ALL_MOVIES:
	_genres = Movie_generes[movie.genre > 0]
	for g in _genres:
		Movie_by_generes[g].append(movie)


if __name__ == "__main__":
	userid = random.randrange(0, 943)
	displayUser(userid, True)
	# kkk = k_similar_users(V, userid, k=10)
	# SU = get_similar_users_from_DB(userid)
	US1, time_graph = greedy_filtering(matrix=V,similarity=cosine_similarity)
	SU  = get_k_similar_users(US1,userid)
	final_reccom = getRecommendations(V, userid, SU)
	print "Movies recommendations for current user, based on his prefrences "
	print final_reccom

	if USE_DB:
		conn.commit()
		conn.close()

# Always last line
