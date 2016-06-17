from recommeder import *
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask import Blueprint, request, render_template, flash, g, session, redirect, url_for
from flask import Flask, request, redirect, url_for
from utils import *

app = Flask(__name__)
# SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(_basedir, 'bogie.db')

userID = 552

@app.route('/')
def home():
    	return render_template('index.html')

@app.route('/movies')
def all_movies():
	limit = int(request.args.get('limit', 100))
	order = int(request.args.get('order', 1))
	_movies = [ x.getInfo() for x in ALL_MOVIES]
	_movies.sort(key=overallRating)
	if order > 0:_movies.reverse()
	_movies = _movies[:limit]
	return render_template('rating.html',movies=_movies)

@app.route('/genres/<genre>')
def genres(genre):
	if genre not in Movie_generes:return redirect('/')

	limit = int(request.args.get('limit', 100))
	order = int(request.args.get('order', 1))
	muvii = [e.getInfo() for e in Movie_by_generes[genre]]
	muvii.sort(key=overallRating)
	if order > 0:muvii.reverse()
	muvii = muvii[:limit]
	return render_template('rating.html',movies= muvii)

@app.route('/movie/<int:movieId>')
def movie_movieId(movieId):
	if movieId >= len(ALL_MOVIES):return redirect('/')

	movie = ALL_MOVIES[movieId]
	return render_template('movieDetails.html',movie=movie)


@app.route('/test')
def test():
	return render_template('test.html',data=MovieNames)

# @app.route('/login/<int:userI>')
# def login_with_userId(userI):
# 	userID = userI
# 	return render_template('index.html',message='Logged in')



@app.route('/user')
def user_details():
	userI = int(request.args.get('userid', userID))
	_names_of_rated, user_string = getUser(userI)
	return render_template('test.html',data=_names_of_rated, datax=user_string)

# Movie_by_generes

if __name__ == "__main__":
    app.run(debug=True)