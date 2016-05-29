from recommeder import *
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask import Blueprint, request, render_template, flash, g, session, redirect, url_for
from flask import Flask, request, redirect, url_for

app = Flask(__name__)
# SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(_basedir, 'bogie.db')








@app.route('/')
def home():
    	return render_template('index.html')

@app.route('/movies')
def movies():
	limit = int(request.args['limit']) or 1682
	_movies = []
	for x in xrange(len(MovieInfo)):
		info = MovieInfo[x]
		_genres = Movie_generes[info['genre']>0]
		_movies.append({'name': M[x].decode('cp1252'),'movieId': x,'overallRating':overall_ranking[x], 'imdbUrl':info['imdbUrl'], 'genres': _genres})
	# return render_template('rating.html')
	return render_template('rating.html',movies=_movies)

@app.route('/rating', methods=['POST'])
def post_rating():
	# return render_template('rating.html')
	return render_template('rating.html',movies=movies)

@app.route('/genres/<genre>')
def genres(genre):
	# limit = int(request.args['limit']) or 1682
	if genre in Movie_generes:
		muvii = [{"name": e['name'],'movieId': e['movieId'],'overallRating':rankings[e['movieId']], 'imdbUrl':e['imdbUrl'], 'genres': Movie_generes[e['genre']>0]} for e in MovieInfo[Movie_by_generes[genre]] ]
		return render_template('rating.html',movies= muvii[:20])


# Movie_by_generes

if __name__ == "__main__":
    app.run(debug=True)