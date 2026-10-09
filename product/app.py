"""Movie product: UI only, consumes platform API. No recommendation logic."""
import os
from flask import Flask, request, render_template, redirect

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Reuse existing app static assets (Materialize, etc.)
STATIC = os.path.join(ROOT, "app", "static")

from product.client import (
    get_all_items,
    get_items_by_ids,
    get_all_users,
    get_recommendations,
    item_to_movie,
)

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")

DEFAULT_USER_ID = 1


def _movies_from_items(items):
    return [item_to_movie(it) for it in items]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/movies")
def all_movies():
    limit = int(request.args.get("limit", 100))
    order = int(request.args.get("order", 1))
    items = get_all_items()
    movies = _movies_from_items(items)
    movies.sort(key=lambda m: m.get("overallRating") or 0, reverse=(order > 0))
    movies = movies[:limit]
    return render_template("rating.html", movies=movies)


@app.route("/genres/<genre>")
def by_genre(genre):
    items = get_all_items()
    movies = _movies_from_items(items)
    filtered = [m for m in movies if genre in (m.get("genres") or [])]
    limit = int(request.args.get("limit", 100))
    order = int(request.args.get("order", 1))
    filtered.sort(key=lambda m: m.get("overallRating") or 0, reverse=(order > 0))
    filtered = filtered[:limit]
    return render_template("rating.html", movies=filtered)


@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
    items = get_items_by_ids([movie_id])
    if not items:
        return redirect("/")
    movie = item_to_movie(items[0])
    return render_template("movieDetails.html", movie=movie)


@app.route("/user")
def user_view():
    user_id = int(request.args.get("userid", DEFAULT_USER_ID))
    try:
        users = get_all_users()
        user_ids = [u["id"] for u in users]
        if user_id not in user_ids:
            user_id = user_ids[0] if user_ids else 1
    except Exception:
        user_id = DEFAULT_USER_ID
    recs = get_recommendations(user_id, limit=10)
    rec_items = get_items_by_ids([r["item_id"] for r in recs]) if recs else []
    rec_movies = _movies_from_items(rec_items)
    user_attrs = {}
    for u in get_all_users():
        if u["id"] == user_id:
            user_attrs = u.get("attributes") or {}
            break
    user_string = "User: %s. Attributes: %s" % (user_id, user_attrs)
    return render_template(
        "user.html",
        user_id=user_id,
        user_string=user_string,
        recommendations=rec_movies,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
