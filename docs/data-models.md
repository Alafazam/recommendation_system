# Data Models: recommendation_system

## Overview

The app uses **in-memory structures** loaded from the MovieLens 100k flat files. Optional **SQLite** is used only to cache user–user (and similar) similarity matrices when `USE_DB=True`.

## In-Memory Structures

### Rating Matrix and Users

- **V** — NumPy array of shape `(943, 1682)` (users × movies). Loaded from `u.data`; optionally TF-IDF normalized in `get_dataset.py` via `baseAlgos.TF_IDF_normalize_ratings`.
- **U** — User metadata array (943 users): `[userid, age, gender, occupation, zipcode]` per row; from `u.user`.

### Movies and Genres

- **ALL_MOVIES** — Array of `Movie` objects (1682 movies). Loaded from `u.item` in `get_dataset.LoadMovies()`; each has `name`, `movieId`, `imdbUrl`, `genre` (binary vector), and rating-related attributes.
- **Movie_generes** — Genre names array; from `u.genre`.
- **Movie_by_generes** — Genre-indexed structure used for `/genres/<genre>` (built from `ALL_MOVIES` and genres).

### Movie Class (`app/utils.py`)

- **Attributes:** `name`, `movieid`, `imdbUrl`, `genre`, `genres`, `ratings`, `ratedBy`, `similar`, `avgRating`, `updated`.
- **Methods:** `calculateAverageRating()`, `addRating()`, `updateRating()`, `getAverageRating()`, `setRatings(V)`, `getInfo()` (and others). Used for per-movie ratings and display.

## Data Sources (MovieLens 100k)

| File | Content |
|------|--------|
| `u.data` | user_id, movie_id, rating, timestamp |
| `u.item` | movie id, title, release date, URL, genre vector (19 bins) |
| `u.user` | user id, age, gender, occupation, zipcode |
| `u.genre` | genre name list |
| `u.occupation` | occupation list |
| `u1.base` / `u1.test` etc. | Train/test splits (used by scripts if needed) |

Default load path is relative to `_basedir` (e.g. `app/ml-100k/`).

## Optional SQLite Schema

When `USE_DB=True` in `app/recommeder.py` / `app/baseAlgos.py`:

- **greedyF_user_similarity_matrix** — `(userID, otherID, similarity)` for greedy-filtered user similarity.
- **user_similarity_matrix** — same shape for full/user similarity matrix.

Tables are created on first run if the DB file is missing. No ORM; raw `sqlite3` in the recommender layer.

## Relationships

- **User–item:** Implicit in `V[user_index, movie_index] = rating`.
- **Movie–genre:** Each movie has a binary genre vector; `Movie_generes` gives names; `Movie_by_generes` groups movies by genre.
- **Similarity:** User–user (and optionally item–item) similarity is computed in memory or read/written from SQLite when `USE_DB=True`.
