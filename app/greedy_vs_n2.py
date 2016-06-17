from recommeder import *
import scipy as sp
import matplotlib.pyplot as plt


print 'Greedy_filtering'
US1, time_graph = greedy_filtering()
print ''



print 'kNN similarity throughN^2 '
US2, time_graph1  = nSquare_user_similarity_matrix()

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

similarUsers_greedy = find_k_similar_users(US1,100)
similarUsers_nSquare = find_k_similar_users(US2,100)


# subplot 1
# plt.subplot(121)
# print time_graph for Greedy_filtering
plt.ylabel("users")
plt.xlabel("Time ( in s)")
x_val_ = [x[1] for x in time_graph]
y_time_ = [x[0] for x in time_graph]
plt.title("Time vs Users For Greedy_filtering and N^2")
plt.scatter(x_val_,y_time_,marker='x')
plt.plot(x_val_,y_time_)
# plt.show()
# plt.autoscale(tight=True)



# plt.subplot(122)
# print time_graph for n^2
x_val_ = [x[1] for x in time_graph1]
y_time_ = [x[0] for x in time_graph1]
plt.scatter(x_val_,y_time_,marker='o',c='g')
plt.plot(x_val_,y_time_)
plt.autoscale()

plt.show()


