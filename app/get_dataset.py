from scipy import spatial
from utils import *
from math import log10
import numpy as np
import os, datetime

from baseAlgos import TF_IDF_normalize_ratings

_basedir = os.path.abspath(os.path.dirname('./'))

# if not __name__ == "__main__":
	# _basedir +="/app"

def LoadGeneres(path='/ml-100k'):
	All_generes = []
	for line in open(_basedir + path+'/u.genre'):
		genre , genreId = line.split('|')
		All_generes.append(genre)
	return np.array(All_generes)


def LoadRatings(path='/ml-100k'):
	ratings = np.zeros((943, 1682)).astype(float)
	for line in open(_basedir + path +'/u.data'):
		userid, movieid, rating, ts = line.split('\t')
		ratings[int(userid) - 1, int(movieid) - 1] = float(rating)
	return np.array(ratings)


def LoadMovies(path='/ml-100k'):
	# Get movie titles, we have 1682 movies
	ALL_MOVIES = []
	for line in open(_basedir + path +'/u.item'):
		data_array = line.rstrip('\n').split('|')
		movieid, title = data_array[0:2]
		genre = np.array(data_array[5:]).astype(int)
		a_movie = Movie({'name': data_array[1],'imdbUrl': data_array[4], 'genre': np.array(genre), 'movieId': int(movieid) -1 })
		ALL_MOVIES.append(a_movie)
	return np.array(ALL_MOVIES)

def LoadUsers(path='/ml-100k'):
	# we have 943 users in format:
	# user id | age | gender | occupation | zip code
	U = []
	for line in open(_basedir + path+'/u.user'):
		userid, age, gender, occupation, zipcode = line.split('|')
		U.append([userid, age, gender, occupation, zipcode])
	return np.array(U)

def LoadOccupations(path='/ml-100k'):
	# u.occupation -- A list of the occupations.
	occupation = []
	for line in open(_basedir + path+'/u.occupation'):
		occupation.append(line)
	return occupation

V_original = LoadRatings()
U = LoadUsers()
ALL_MOVIES = LoadMovies()
Movie_generes = LoadGeneres()

V = TF_IDF_normalize_ratings(V_original)
