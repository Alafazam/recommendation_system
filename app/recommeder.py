from scipy import spatial
import numpy as np
from math import sqrt
import os, datetime
import time,json
import random
import sqlite3


db = 'MovieLens.db'
USE_DB = True
if USE_DB :
	if USE_DB and not os.path.isfile(db): crt = True
	else: crt = False
	conn = sqlite3.connect(db)
	c = conn.cursor()
	if crt:	c.execute('''CREATE TABLE greedyF_user_similarity_matrix(userID real, otherID real, similarity real)''')

_basedir = os.path.abspath(os.path.dirname('./'))
# if not __name__ == "__main__":
# 	_basedir +="/app"


def log(x):
	for q in range(len(x)):
			print x[q]
np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')
cosine_similarity = spatial.distance.cosine


def loadMovieLens(path='/ml-100k'):
	# Get movie titles, we have 1682 movies
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
	# we have 943 users
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

def greedy_filtering(matrix = V, n=0, rho = 10, measure_time=True, minimum_similarity = 0.3, similarity = cosine_similarity, USE_DB = False, top_k_values = 96):
	"""
	rho is the number of elements in the intersecting arrays
	"""
	if n == 0: n = matrix.shape[0]
	rho = 10
	range_of_X = n
	similarity_matrix =  np.zeros(shape=(n,n))
	auxillary_matrix = [ [] for i in xrange(n)]
	for i in xrange(n):
		auxillary_matrix[i] = (np.argsort(matrix[i]))[::-1]

	if measure_time:
		t_graph = []
		start = time.time()

	for x in xrange(0,range_of_X):
		if measure_time and x%100 == 0 :
			print x, time.time()-start, 's'

		if np.count_nonzero(matrix[x]) == 0:
			continue
		for y in xrange(x+1,range_of_X):
			if np.count_nonzero(matrix[y]) == 0:
				continue
			dataSetI  = auxillary_matrix[x][:top_k_values]
			dataSetII = auxillary_matrix[y][:top_k_values]
			intersection = np.intersect1d(dataSetI,dataSetII)
			array_I = matrix[x]
			array_II = matrix[y]
			# print intersection
			if len(intersection) > rho:
				similarity_score = 1 - similarity(array_I, array_II)
				if similarity_score > minimum_similarity:
					if USE_DB :
						c.execute("INSERT INTO 'greedyF_user_similarity_matrix' VALUES (%d, %d, %f)"%(x, y, similarity_score))
						c.execute("INSERT INTO 'greedyF_user_similarity_matrix' VALUES (%d, %d, %f)"%(y, x, similarity_score))
						if x%50 == 0: conn.commit()
					t_graph.append([x,time.time()-start])
					similarity_matrix[x][y] = similarity_score
					similarity_matrix[y][x] = similarity_score

	if measure_time:
		end = time.time()
		print "Total: ",end-start,"s"
		return similarity_matrix , t_graph

	return similarity_matrix


def nSquare_user_similarity_matrix(matrix = V, measure_time=True, similarity= cosine_similarity, n=0, minimum_similarity = 0.3, USE_DB = False):
	""" For n = 100 Avg time is 8.7 secs.
		For n = 200 Avg time is 32.14 secs.
		For n = 943 Time is too much ~900secs.
	"""
	print 'start'
	if n == 0: n = matrix.shape[0]

	if measure_time:
		t_graph = []
		start = time.time()
	similarity_matrix = np.zeros(shape=(n,n))

	for userid in range(n):
		if measure_time and userid%100 == 0:
			print userid, time.time()-start, 'secs.'

		for other in range(userid + 1, n):
			# intersection = np.intersect1d(np.nonzero(matrix[userid]),np.nonzero(matrix[other]))
			# if len(intersection) < 10: continue
			userN = matrix[userid]
			otherN = matrix[other]
			similarity_score = 1 - similarity(userN, otherN)
			if USE_DB :
				c.execute("INSERT INTO 'user_similarity_matrix' VALUES (%d, %d, %f)"%(userid, other, similarity_score))
				c.execute("INSERT INTO 'user_similarity_matrix' VALUES (%d, %d, %f)"%(other, userid, similarity_score))
				if userid%50 == 0: conn.commit()
			if similarity_score > minimum_similarity:
				similarity_matrix[userid][other] = similarity_score
				similarity_matrix[other][userid] = similarity_score
				t_graph.append([userid,time.time()-start])
	if measure_time:
		print "Total Time Elapsed: ",time.time()-start,"s"
		return similarity_matrix , t_graph
	return similarity_matrix


# US = nSquare_user_similarity_matrix()


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


def k_similar_users(ratings, userid, k=10, similarity = cosine_similarity):
	if k == 0: k = 943
	scores=[[similarity(ratings[userid], ratings[other]), other] for other in range(0,943) if other!=userid]
	# Sort the list so the highest scores appear at the top
	scores.sort()
	scores.reverse()
	return scores[0: k]

# kkk = k_similar_users(V, userid, k=10)
# print "userids of similar Users to selected user and their similarity score: "

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

	if USE_DB:
		conn.commit()
		conn.close()

# Always last line
