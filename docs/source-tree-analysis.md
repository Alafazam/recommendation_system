# Source Tree Analysis

## Annotated Directory Tree

```
recommendation_system/
├── server.py              # Entry point: runs Flask app (app.run(debug=True))
├── run.py                 # Standalone script: load data, pick user, print recommendations (CLI)
├── mlens.py               # Alternate entry: uses scripts/sim_dist, loads dataset, item/user recommendations
├── README.md              # Project title only
├── LICENSE
├── MovieLens.db           # Optional SQLite DB (similarity cache when USE_DB=True)
│
├── app/                   # Main application package (Flask + recommendation logic)
│   ├── __init__.py        # Flask app, routes: /, /movies, /genres/<genre>, /movie/<id>, /test, /user
│   ├── recommeder.py      # Core recommendation: getRecommendations, similar users, similarity matrix, optional DB
│   ├── similarityMeasures.py  # cosine_similarity, pearson_similarity (SciPy/custom)
│   ├── baseAlgos.py       # greedy_filtering, nSquare_user_similarity_matrix, TF_IDF_normalize_ratings
│   ├── get_dataset.py     # Load MovieLens 100k: V, U, ALL_MOVIES, Movie_generes (from ml-100k/)
│   ├── utils.py           # Movie class, overallRating, genre loading, rating helpers
│   ├── get_dataset.py     # Dataset loading (path defaults to /ml-100k relative to basedir)
│   ├── accuracy_test.py   # Accuracy/testing utilities
│   ├── greedy_vs_n2.py    # Greedy vs N² similarity comparison
│   ├── RCF_test.py        # Recommendation tests
│   ├── tfidf_test.py      # TF-IDF tests
│   ├── test.py, newtest.py
│   ├── rating.html        # Fragment or alternate template
│   ├── templates/         # Jinja2 HTML templates
│   │   ├── index.html     # Home: “Demo Movie Recommender”, Get Started
│   │   ├── movieDetails.html
│   │   ├── rating.html    # Movie list (e.g. /movies, /genres/<genre>)
│   │   └── test.html      # Test/data view
│   ├── static/            # CSS, JS, images (e.g. materialize.min.css, 764+ PNGs)
│   └── ml-100k/           # MovieLens 100k data files
│       ├── u.data         # Ratings (user, movie, rating, timestamp)
│       ├── u.item         # Movies (id, title, …)
│       ├── u.user         # Users (id, age, gender, occupation, zipcode)
│       ├── u.genre        # Genre list
│       ├── u.occupation   # Occupation list
│       └── u1.base, u1.test, … # Train/test splits
│
└── scripts/               # Standalone/legacy scripts
    ├── pearson.py
    ├── sql_test.py
    └── old/               # Legacy: app.py, cluster.py, recommendations.py, load_data.py, etc.
```

## Critical Folders Summary

| Path           | Purpose |
|----------------|---------|
| `app/`         | Flask app, routes, recommendation engine, dataset loading, templates, static assets. |
| `app/templates/` | Server-rendered pages: home, movie list, movie details, test view. |
| `app/static/`  | Front-end assets (CSS, JS, images). |
| `app/ml-100k/` | MovieLens 100k dataset (required for V, U, ALL_MOVIES). |
| `scripts/`     | Standalone and legacy scripts (not part of the web server). |

## Entry Points

- **Web:** `server.py` → `from app import app; app.run(debug=True)`.
- **CLI / one-off:** `run.py` (imports from `app`, loads data, prints recommendations for a random user).
- **Alternative CLI:** `mlens.py` (uses `sim_dist` from scripts, loads dataset, prints user/item recommendations).

## Integration Notes

- Single process: no separate API service or front-end build. All logic and UI live in `app/`.
- Data is loaded at import time from `app/get_dataset.py` (paths relative to `_basedir`; default `/ml-100k`). Ensure `app/ml-100k/` (or configured path) is present.
