from scipy import spatial
from math import sqrt
from math import log10


def cosine_similarity(v1,v2):
 return 1 - spatial.distance.cosine(v1,v2)

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

def dot_product(v1, v2):
	return sum(map(lambda x: x[0] * x[1], izip(v1, v2)))

def cosine_measure(v1, v2):
	prod = dot_product(v1, v2)
	len1 = math.sqrt(dot_product(v1, v1))
	len2 = math.sqrt(dot_product(v2, v2))
	return prod / (len1 * len2)
