from math import sqrt
import os, datetime
import time,json
import csv

from matplotlib import pyplot as plt
import numpy as np
import random

_basedir = os.path.abspath(os.path.dirname(__file__))




def loadMovieLensLinks(path='/ml-20m'):
	# Get movie imdb and tmdb links
	moviesLinks={}
	with open(_basedir + path + '/links.csv', 'rb') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		next(csvreader)
		for row in csvreader:
			(id,imdb,tmdb)=row
			moviesLinks[id]={"id":id,"imdb":imdb,"tmdb":tmdb}
			print row
	return moviesLinks

# p = loadMovieLensLinks()
# q = np.array(p)


def loadMovieLensTitles(path='/ml-20m'):
	# Get movie titles
	moviesTitles={}
	with open(_basedir + path + '/movies.csv', 'rb') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		next(csvreader)
		for row in csvreader:
			(id,title,genres)=row
			genres = genres.split('|')
			moviesTitles[id]={"id":id,"title":title,"genre":genres}
			# print moviesTitles[id]
	return moviesTitles

# p = loadMovieLensTitles()
# q = np.array(p)
# print q


def loadMovieLensRatings(path='/ml-20m'):
	# Get movie titles
	moviesRatings=[]
	with open(_basedir + path + '/ratings.csv', 'rb') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		next(csvreader)
		for row in csvreader:
			print row
			# (userid,movieId,rating)=row
			# moviesRatings.append({"userid":userid,"movieId":movieId,"rating":rating})
			# print moviesTitles[id]
	return moviesRatings

p = loadMovieLensRatings()[0:500]
q = np.array(p)
print q
