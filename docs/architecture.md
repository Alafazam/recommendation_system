# Architecture: recommendation_system

## Executive Summary

The project is a **monolithic Python web application**: a Flask server that serves server-rendered HTML and implements a collaborative-filtering movie recommender. Recommendations are computed in-process using NumPy/SciPy over the MovieLens 100k dataset; optional SQLite stores precomputed user similarity for performance.

## Technology Stack

- **Language:** Python 2.x
- **Web framework:** Flask (routes, Jinja2 templates, debug server)
- **Data/compute:** NumPy, SciPy (similarity metrics)
- **Persistence:** In-memory arrays by default; optional SQLite (MovieLens.db) for similarity cache
- **Front-end:** Server-rendered HTML, Materialize CSS

## Architecture Pattern

- **Request/response web app:** One process handles HTTP and recommendation logic.
- **Layered within app:** Routes → business logic (recommender, similarity) → data (get_dataset, utils, optional DB).
- No message queues, no separate API service; optional DB is used only for caching similarity matrices.

## Data Architecture

- **Primary:** In-memory NumPy arrays: user–item rating matrix `V`, user metadata `U`, movie list `ALL_MOVIES`, genre list `Movie_generes`. Loaded from flat files in `app/ml-100k/`.
- **Optional:** SQLite `MovieLens.db` for storing user–user similarity (when `USE_DB=True` in recommender/baseAlgos).
- **Domain model:** `Movie` class in `utils.py` (name, movieid, genre, ratings, etc.); no ORM in use by default.

## API Design

All “API” surface is **browser-oriented Flask routes** (HTML responses):

- `GET /` — Home (index)
- `GET /movies` — All movies list (optional `limit`, `order`)
- `GET /genres/<genre>` — Movies by genre (optional `limit`, `order`)
- `GET /movie/<int:movieId>` — Movie details
- `GET /test` — Test view (e.g. movie names)
- `GET /user?userid=<id>` — User view (rated movies, user info)

See [API Contracts](./api-contracts.md) for details.

## Component Overview

- **Routes:** `app/__init__.py` — defines all URLs and renders templates.
- **Recommendation engine:** `app/recommeder.py`, `baseAlgos.py`, `similarityMeasures.py` — similarity computation, filtering, recommendations.
- **Data loading:** `app/get_dataset.py`, `utils.py` — load ML-100k files and expose `V`, `U`, `ALL_MOVIES`, `Movie_generes`.
- **UI:** `app/templates/*.html`, `app/static/` — server-rendered pages and assets.

## Source Tree

See [Source Tree Analysis](./source-tree-analysis.md) for an annotated directory tree and entry points.

## Development Workflow

- Run: `python server.py` (or equivalent Flask run).
- No build step; no separate front-end toolchain. Dataset must be present under `app/ml-100k/` (or path configured in get_dataset).

See [Development Guide](./development-guide.md) for setup and commands.

## Deployment Architecture

- No deployment config in repo (no Dockerfile, docker-compose, or CI/CD). Single process, development server only.

## Testing Strategy

- Test/accuracy scripts present: `accuracy_test.py`, `greedy_vs_n2.py`, `RCF_test.py`, `tfidf_test.py`, `test.py`, `newtest.py`. No standardized test runner or CI configuration observed.
