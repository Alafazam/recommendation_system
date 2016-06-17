import numpy as np
import os, datetime

_basedir = os.path.abspath(os.path.dirname('./'))

# if not __name__ == "__main__":
	# _basedir +="/app"


def overallRating(x):
	return x['overallRating']



def loadGeneres(path='/ml-100k'):
	All_generes = []
	for line in open(_basedir + path+'/u.genre'):
		genre , genreId = line.split('|')
		All_generes.append(genre)
	return np.array(All_generes)

Movie_generes = loadGeneres()




class Movie(object):
	"""docstring for Movie"""
	def __init__(self, arg):
		self.name = arg['name']
		self.movieid = arg['movieId']
		self.imdbUrl = arg['imdbUrl']
		self.genre = arg['genre']
		self.genres = Movie_generes[self.genre > 0]
		self.ratings = []
		self.ratedBy = []
		self.similar = []
		self.avgRating = 0
		self.updated = 0

	def calculateAverageRating(self):
		self.avgRating = self.ratings.mean()

	def addRating(self, userid, rating):
		if rating==0:return
		self.updated = 1
		self.ratedBy.append(int(userid))
		self.ratings.append(float(rating))

	def updateRating(self, userid, rating):
		self.updated = 1
		index = self.ratedBy.index(userid)
		self.ratings[index] = rating

	def getAverageRating(self):
		if self.updated: self.calculateAverageRating()
		return self.avgRating

	def setRatings(self,V):
		self.ratings = np.array([ V[x][self.movieid] for x in range(V.shape[0]) if V[x][self.movieid]!=0])
		self.avgRating = self.ratings.mean()
		self.ratedBy = np.array([ x for x in range(V.shape[0]) if V[x][self.movieid]!=0 ] )

	def __str__(self):
		return "Name: %s, \nAverage Rating: %.2f, \nMovieId: %d " % (self.name, self.avgRating, self.movieid)

	def __repr__(self):
		return "Name: %s, Average Rating: %.2f, Movie ID: %d " % (self.name, self.avgRating, self.movieid)

	def getInfo(self):
		return {'name': self.name, 'movieId': self.movieid, 'overallRating': self.avgRating, 'imdbUrl': self.imdbUrl, 'genres': self.genres}

