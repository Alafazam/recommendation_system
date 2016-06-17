from recommeder import *
import scipy as sp
import matplotlib.pyplot as plt
from get_dataset import V, U, ALL_MOVIES, Movie_generes, V_original
from baseAlgos import custom_greedy_filtering, greedy_filtering, nSquare_user_similarity_matrix

V2 = V.reshape(V.shape[1],V.shape[0]).copy()

US1, Q = greedy_filtering(matrix=V2, similarity=cosine_similarity)

matrix=V_original
userID=552
k=30
# def RCF(matrix, userID, Q, k=30):
ratings_by_user = np.array(matrix[userID])
I_R = [ i for i in range(len(ratings_by_user)) if ratings_by_user[i] > 0]
I_U = [ i for i in range(len(ratings_by_user)) if ratings_by_user[i] == 0]

S = [ [] for i in range(len(ratings_by_user))]

for i in range(len(ratings_by_user)):
	if i in I_R:
		for sim_j, j in Q[i]:
			print i, j , sim_j
			S[j].append([sim_j,i])
	if i in I_U:
		S[i].sort()
		S[i] = S[i][::-1]
		if len(S[i]) > k:
			S[i] = S[i][:k]
# return S

# S = [ i for i in S if len(i)>0]
# S = RCF(matrix=V_original,userID=552,Q=Q)

P = [ 0 for i in range(len(ratings_by_user))]

for i in range(len(S)):
	if len(S[i])==0:continue

	average_rating = ALL_MOVIES[i].getAverageRating()
	aSum = 0.0
	for simi, n in S[i]:
		aSum +=   (V_original[userID][i] - ALL_MOVIES[n].getAverageRating())*simi
	P[i] = aSum

P = np.argsort(P)
P = P[::-1]
P = P[:20]

for mov in P:
	print ALL_MOVIES[mov]
