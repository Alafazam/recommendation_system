import numpy as np
from utils import *
from math import log10
from similarityMeasures import cosine_similarity,pearson_similarity
import time

def custom_greedy_filtering(matrix, n = 0, rho = 50, measure_time=True, minimum_similarity = 0.3, similarity = cosine_similarity, USE_DB = False, top_k_values = 96):
	"""
	rho is the number of elements in the intersecting arrays
	"""
	if n == 0: n = matrix.shape[0]
	rho = 10
	range_of_X = n
	similarity_matrix =  np.zeros(shape=(n,n))
	auxillary_matrix = [ set((np.argsort(matrix[i]))[::-1][:top_k_values]) for i in xrange(n)]
	if measure_time:
		t_graph = []
		start = time.time()
	for x in xrange(0,range_of_X):
		if measure_time and x%100 == 0 :
			print x, time.time()-start, 's'
			t_graph.append([x,time.time()-start])
		# if np.count_nonzero(matrix[x]) == 0:continue
		for y in xrange(x+1,range_of_X):
			# if np.count_nonzero(matrix[y]) == 0:continue
			dataSetI  = auxillary_matrix[x]
			dataSetII = auxillary_matrix[y]
			intersection = dataSetI.intersection(dataSetII)
			array_I = matrix[x]
			array_II = matrix[y]
			# print intersection
			if len(intersection) > rho:
				similarity_score = similarity(array_I, array_II)
				if similarity_score > minimum_similarity:
					if USE_DB :
						c.execute("INSERT INTO 'greedyF_user_similarity_matrix' VALUES (%d, %d, %f)"%(x, y, similarity_score))
						c.execute("INSERT INTO 'greedyF_user_similarity_matrix' VALUES (%d, %d, %f)"%(y, x, similarity_score))
						if x%50 == 0: conn.commit()
					similarity_matrix[x][y] = similarity_score
					similarity_matrix[y][x] = similarity_score

	if measure_time:
		end = time.time()
		print "Total: ",end-start,"s"
		return similarity_matrix , t_graph

	return similarity_matrix


def nSquare_user_similarity_matrix(matrix, measure_time=True, similarity= cosine_similarity, n=0, minimum_similarity = 0.3, USE_DB = False):
	""" With pearson similarity
		For n = 100 Avg time is 8.7 secs.
		For n = 200 Avg time is 32.14 secs.
		For n = 943 Time is too much ~900secs.
	"""
	print 'nSquare_user_similarity_matrix'
	if n == 0: n = matrix.shape[0]
	if measure_time:
		t_graph = []
		start = time.time()
	similarity_matrix = np.zeros(shape=(n,n))

	for userid in range(n):
		if measure_time and userid%100 == 0 and similarity == cosine_similarity:
			print userid, time.time()-start, 's.'
			t_graph.append([userid,time.time()-start])

		if measure_time and userid%20 == 0 and similarity == pearson_similarity :
			print userid, time.time()-start, 's.'
			t_graph.append([userid,time.time()-start])

		for other in range(userid + 1, n):
			# intersection = np.intersect1d(np.nonzero(matrix[userid]),np.nonzero(matrix[other]))
			# if len(intersection) < 10: continue
			userN = matrix[userid]
			otherN = matrix[other]
			similarity_score = similarity(userN, otherN)
			if USE_DB :
				c.execute("INSERT INTO 'user_similarity_matrix' VALUES (%d, %d, %f)"%(userid, other, similarity_score))
				c.execute("INSERT INTO 'user_similarity_matrix' VALUES (%d, %d, %f)"%(other, userid, similarity_score))
				if userid%50 == 0: conn.commit()
			if similarity_score > minimum_similarity:
				similarity_matrix[userid][other] = similarity_score
				similarity_matrix[other][userid] = similarity_score
	if measure_time:
		print "Total Time Elapsed: ",time.time()-start,"s"
		return similarity_matrix , t_graph
	return similarity_matrix


def TF_IDF_normalize_ratings(matrix):
	F_uj = [ log10(matrix.shape[0] / (np.count_nonzero(matrix[i]) + 1)) for i in xrange(matrix.shape[0]) ]
	# print F_uj
	M_i_j = matrix.reshape((1682,943)).copy()
	number_of_users = M_i_j.shape[1]
	number_of_items = M_i_j.shape[0]
	for i in xrange(number_of_items):
		max_i_k  = 1
		for k in xrange(number_of_users):
			if M_i_j[i][k] > max_i_k:
				max_i_k = M_i_j[i][k]
		for j in xrange(number_of_users):
			if M_i_j[i][j] == 0:continue
			y = (0.5 * M_i_j[i][j] / max_i_k)  + 0.5
			M_i_j[i][j] = y * F_uj[j]
	return M_i_j.reshape((943,1682))


def genrate_auxilary_matrix(matrix):
	n = matrix.shape[0]
	auxillary_matrix = [ np.argsort(matrix[i])[::-1] for i in xrange(n)]
	return auxillary_matrix

def greedy_filtering(matrix,rho = 300, measure_time=True, minimum_similarity = 0.3, similarity = cosine_similarity):
	print 'Greedy Filtering'
	pK = rho
	n_V = matrix.shape[0]
	n_D = matrix.shape[1]
	auxmatrix = genrate_auxilary_matrix(matrix)
	L = [[] for i in range(n_D)]
	P = [ 0 for i in range(n_V)]
	SC = [ 0 for i in range(n_V)]
	changed = 1

	for vi in range(n_V):
		x = auxmatrix[vi][0]
		L[x].append(vi)
		P[vi] = 1

	start = time.time()
	iterations = 0
	while changed:
		iterations +=1
		changed = 0
		for di in range(n_D):
			for v in L[di]:
				SC[v] = SC[v] + len(L[di])
		for vi in range(n_V):
			if SC[vi] < pK and len(auxmatrix[vi]) > P[vi]:
				x = auxmatrix[vi][P[vi]]
				L[x].append(vi)
				changed = 1
				P[vi] = P[vi] + 1
		print iterations, time.time()-start, 's.'

	similarity_matrix = np.zeros(shape=(n_V,n_V))
	Q = [ [] for vi in range(n_V)]
	for di in range(n_D):
		for x in L[di]:
			for y in L[di]:
				if x==y:continue
				similarity_score = similarity(matrix[x],matrix[y])
				if similarity_score > minimum_similarity:
					Q[x].append([similarity_score,y])
					similarity_matrix[x][y] = similarity_score
	for vi in range(n_V):
		Q[vi].sort()
	return similarity_matrix, Q


