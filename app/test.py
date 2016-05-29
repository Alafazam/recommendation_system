from recommeder import *
from scipy import spatial
import math
from itertools import izip

def dot_product(v1, v2):
    return sum(map(lambda x: x[0] * x[1], izip(v1, v2)))

def cosine_measure(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)


rho = 10
V2 = V.reshape(1682,943)
dimenstion_x = 1682

print V2.shape
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
        # print dataSetII
        # print V2[x]
        if len(intersection) > rho:
            cosi = 1 - spatial.distance.cosine(V2[x][intersection], V2[y][intersection])
            if cosi > minimum_similarity:
                # print x, y, cosi
                V4[x].append(y)
                # V5[x].append(cosi)
            	V5.append([x,y,cosi])
    # print x, len(V4[x]), V5[x]
