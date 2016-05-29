import numpy as np
from math import sqrt
import os, datetime
import time,json
import random

import sqlite3
db = 'MovieLens.db'
if not os.path.isfile(db): crt = True
else: crt = False
conn = sqlite3.connect(db)
c = conn.cursor()
if crt:	c.execute('''CREATE TABLE user_similarity_matrix(userID real, otherID real, similarity real)''')

_basedir = os.path.abspath(os.path.dirname('./'))
# if not __name__ == "__main__":
# 	_basedir +="/app"


def log(x):
	for q in range(len(x)):
			print x[q]
np.set_printoptions(suppress=True)


def loadMovieLens(path='/ml-100k'):
	# Get movie titles
	M = ["" for i in range(0,1682)]
	MovieInfo = ["" for i in range(0,1682)]
	M_genere = np.ones((1682,19)).astype(int)

	for line in open(_basedir + path +'/u.item'):
		data_array = line.rstrip('\n').split('|')
		movieid, title = data_array[0:2]
		# date = data_array[2]
		genre = np.array(data_array[5:]).astype(int)
		M_genere[int(movieid) - 1] = genre
		MovieInfo[int(movieid) - 1] = {'name': data_array[1].decode('cp1252'),'imdbUrl': data_array[4], 'genre': np.array(genre), 'movieId': int(movieid) -1 }
		M[int(movieid) - 1] = str(title)
	# Load data
	V = np.zeros((943, 1682)).astype(int)
	for line in open(_basedir + path +'/u.data'):
		userid, movieid, rating, ts = line.split('\t')
		V[int(userid) - 1, int(movieid) - 1] = int(rating)
	# user id | age | gender | occupation | zip code
	U = ["" for i in range(0,943)]
	for line in open(_basedir + path+'/u.user'):
		userid, age, gender, occupation, zipcode = line.split('|')
		U[int(userid) - 1] = [userid, age, gender, occupation, zipcode]
	# u.occupation -- A list of the occupations.
	Occu = ["" for i in range(0,21)]
	i=0
	for line in open(_basedir + path+'/u.occupation'):
		occupations = line
		Occu[i] = occupations
		i+=1
	All_generes = []
	for line in open(_basedir + path+'/u.genre'):
		genre , genreId = line.split('|')
		All_generes.append(genre)

	return V, np.array(M), np.array(U), np.array(Occu), np.array(M_genere) , np.array(MovieInfo), np.array(All_generes)


V, M, U, O, M_genere, MovieInfo, Movie_generes = loadMovieLens()

# userid = random.randrange(0, 943)


def getUser(id,log=False):
	id = id-1
	mrated = V[id]
	names = np.ones(1682).astype(int)
	for x in range(len(mrated)):
		if mrated[int(x)] == 0 : names[x] = 0
	_names_of_rated = M[names.nonzero()]
	user = U[id]
	user_string = ("User       : " + str(id+1) + "\n"
              "Age        : " + str(user[1]) + "\n"
              "Gender     : " + str(user[2]) + "\n"
              "Occupation : " + str(user[3]) + "\n"
              "Zipcode    : " + str(user[4]) + ""
              )
	if(log): print user_string
	# print  str(_names_of_rated)
	_names_of_rated = _names_of_rated[:10]
	print "Movies rated by selected user "
	print "\n".join(_names_of_rated)
	print "    "

# getUser(userid, True)



def greedy_filtering():
	"""
	rho is the number of elements in the intersection
	we have An array of ratings in V, we reshape it for item/movies to get V2
	then we argsort each movies's rating and store it in V3
	Then we take top 50 from it and find intersection
	then at last we find cosine similarity in cmmon elements

	"""


	rho = 10
	V2 = V.reshape(1682,943)
	dimenstion_x = 1682
	# print V2.shape
	V3 = ['' for z in xrange(dimenstion_x)]
	for i in xrange(dimenstion_x):
	    sorteda = (np.argsort(V2[i]))[::-1]
	    V3[i] = sorteda

	range_of_users = 1682
	minimum_similarity = 0.3
	V4 = [ [] for p in xrange(range_of_users)]
	V5 = [ [] for p in xrange(range_of_users)]

	for x in xrange(range_of_users-1):
	    if np.count_nonzero(V2[x]) == 0:
	        continue
	    for y in xrange(x+1, range_of_users):
	        if np.count_nonzero(V2[y]) == 0:
	            continue
	        dataSetI  = V3[x][:50]
	        dataSetII = V3[y][:50]
	        # print len(np.intersect1d(V3[x][:100],V3[y][:100])) , x , y
	        intersection = np.intersect1d(dataSetI,dataSetII)
	        if len(intersection) > rho:
	            cosi = 1 - spatial.distance.cosine(V2[x][intersection], V2[y][intersection])
	            if cosi > minimum_similarity:
	                # print x, y, cosi
	                V4[x].append(y)
	                # V5[x].append(cosi)
	            	V5.append([x,y,cosi])
	    # print x, len(V4[x]), V5[x]








def rated_rankings():
	V2 = V.reshape(1682,943)
	# V3 = np.sum(V2,axis=1)
	arv = []
	maxz = -1
	imaxz = -1
	for x in xrange(0,1682):
		asum = float(np.sum(V2[x]))
		n = float(len(V2[x].nonzero()[0]))
		if n != 0: rat = asum/n
		else: rat = 0
		arv.append(rat)
		if rat > maxz:
			maxz = rat
			imaxz = x
	# print arv[imaxz]
	arv = np.array(arv)
	# print np.amax(arv)
	M2 = np.around(arv,decimals=3)
	R = arv.argsort()
	M3 = M2[R][::-1]
	# for x in xrange(0, 1682):
	# 	M[x] = M[x] + "  |  " + str(M2[x])
	return M3, M[R][::-1], R, M2

overall_ranking, M , R, rankings = rated_rankings()
MovieInfo = MovieInfo[R][::-1]


def pearson_similarity(rating1, rating2):
	sum_xy = 0.0
	sum_x = 0.0
	sum_y = 0.0
	sum_x2 = 0.0
	sum_y2 = 0.0
	n = 0
	# print rating1.nonzero()
	for key in rating1.nonzero()[0]:
		if key in rating2.nonzero()[0]:
			n += 1
			x = rating1[key]
			y = rating2[key]
			sum_xy += x*y
			sum_x += x
			sum_y += y
			sum_x2 += x**2
			sum_y2 += y**2
	#if no ratings are in common, we should return 0
	if n == 0:
		return 0
	#now denominator
	denominator = sqrt(sum_x2 - (sum_x**2) / n) * sqrt(sum_y2 - (sum_y**2) / n)
	if denominator == 0:
		return 0
	else:
		return (sum_xy - (sum_x * sum_y) / n) / denominator


def k_similar_users(ratings, userid, k=10, similarity = pearson_similarity):
	if k == 0: k = 943
	scores=[[similarity(ratings[userid], ratings[other]), other] for other in range(0,943) if other!=userid]
	# Sort the list so the highest scores appear at the top
	scores.sort()
	scores.reverse()
	return scores[0: k]

# kkk = k_similar_users(V, userid, k=10)
# print "userids of similar Users to selected user and their similarity score: "

def calulate_user_similarity_matrix(V=V, similarity= pearson_similarity, n=0):
	""" For n = 100 Avg time is 8.7 secs.
		For n = 200 Avg time is 32.14 secs.
		For n = 943 Time is too much ~900secs.
	"""
	start = time.clock()
	if n == 0: n = V.shape[0]
	US = np.zeros((n,n))
	for userid in range(n):
		print userid, time.clock()-start
		for other in range(userid + 1, n):
			score = similarity(V[userid], V[other])
			# print "score bw user %d and %d is %f" % (userid, other,score)
			c.execute("INSERT INTO 'user_similarity_matrix' VALUES (%d, %d, %f)"%(userid, other, score))
			c.execute("INSERT INTO 'user_similarity_matrix' VALUES (%d, %d, %f)"%(other, userid, score))
			US[userid, other] = score
			US[other, userid] = score
		if userid%50 == 0: conn.commit()
	end = time.clock()
	print "Total: ",end-start,"s"
	return US


# US = calulate_user_similarity_matrix()

def get_similar_users(userid,k=50):
	c.execute("SELECT * FROM user_similarity_matrix WHERE userID=%d AND otherID!=%d ORDER BY similarity DESC LIMIT %d;"%(userid, userid, k))
	similarUsers = c.fetchall()
	similarUsers = np.array(similarUsers)
	similarUsers = np.delete(similarUsers, np.s_[0:1], 1)
	return similarUsers

# SU = get_similar_users(552)
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
	return M[final][:10]

# movies by genres
Movie_by_generes = {}
for w in xrange(len(Movie_generes)):
	Movie_by_generes[Movie_generes[w]] = []

for movie in xrange(len(MovieInfo)):
	_genres = Movie_generes[MovieInfo[movie]['genre']>0]
	for g in _genres:
		Movie_by_generes[g].append(MovieInfo[movie]['movieId'])

def dot_product(v1, v2):
    return sum(map(lambda x: x[0] * x[1], izip(v1, v2)))

def cosine_measure(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)














if __name__ == "__main__":
	userid = random.randrange(0, 943)
	getUser(userid, True)
	kkk = k_similar_users(V, userid, k=10)
	SU = get_similar_users(552)
	final_reccom = getRecommendations(V, userid, SU)
	print "Movies recommendations for current user, based on his prefrences "
	print "\n".join(final_reccom)

# Always last line
conn.commit()
conn.close()
