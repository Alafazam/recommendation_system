from recommeder import *
import scipy as sp
import matplotlib.pyplot as plt
from get_dataset import V, U, ALL_MOVIES, Movie_generes
from baseAlgos import custom_greedy_filtering, greedy_filtering, nSquare_user_similarity_matrix


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


# US1, time_graph = custom_greedy_filtering(matrix=V, similarity=cosine_similarity)
US1, Q = greedy_filtering(matrix=V, similarity=cosine_similarity)
US2, time_graph1  = nSquare_user_similarity_matrix(matrix=V, similarity=cosine_similarity)

similarUsers_greedy = find_k_similar_users(US1,10)
similarUsers_nSquare = find_k_similar_users(US2,100)
# # print similarUsers_greedy
# # print len(similarUsers_greedy)
# # print similarUsers_nSquare

performance = []
for i in xrange(len(similarUsers_nSquare)):
	x = similarUsers_greedy[i]
	y = similarUsers_nSquare[i]
	count = 0
	x_ids = [ a[1] for a in x]
	y_ids = [ a[1] for a in y]
	if len(y_ids)==0:continue
	r = [ s for s in x_ids if s in y_ids]
	items_r = len(r)
	items_x = len(x_ids)
	if items_x == 0:items_x=1
	print len(r),len(y_ids),len(x_ids)
	percent = float(items_r)/items_x
	performance.append(percent)


performance_sum = np.array(performance)
# print performance_sum
print 'Accuracy ',performance_sum.mean()*100, '%'

# # matrix = V, n=0, rho = 50, measure_time=True, minimum_similarity = 0.3, similarity = cosine_similarity, USE_DB = False, top_k_values = 96)
