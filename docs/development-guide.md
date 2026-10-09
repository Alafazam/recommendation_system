# Development Guide: recommendation_system

## Prerequisites

- **Python 2.x** (code uses `print` without parentheses, `xrange`, etc.)
- **Pip** (or equivalent) for installing dependencies

## Dependencies

Install manually or via a virtualenv; no `requirements.txt` in repo. Observed imports:

- **Flask** — web server and routes
- **flask_sqlalchemy** — imported in `app/__init__.py` (DB usage is commented out; SQLite used via `sqlite3` in recommender when `USE_DB=True`)
- **NumPy** — rating matrix and array operations
- **SciPy** — `spatial.distance.cosine` in `similarityMeasures.py`

Suggested install:

```bash
pip install Flask numpy scipy
```

(Add `flask_sqlalchemy` if you enable DB-backed features.)

## Environment Setup

- No `.env` or config files required for basic run.
- **Dataset:** MovieLens 100k files must be present. Default path is relative to app’s `_basedir` (e.g. `app/ml-100k/` with `u.data`, `u.item`, `u.user`, `u.genre`, `u.occupation`). Adjust paths in `get_dataset.py` if you place data elsewhere.

## Local Development

**Run the web app:**

```bash
python server.py
```

Flask will start in debug mode (default port 5000). Open http://localhost:5000 in a browser.

**CLI-style recommendation (no server):**

```bash
python run.py
```

Uses a random user and prints recommendations (requires `app` and dataset to be loadable).

**Alternative CLI (scripts path):**

```bash
python mlens.py
```

Uses `scripts` (e.g. `sim_dist`) and prints user/item recommendations.

## Build Process

None. This is an interpreted Python app with no front-end build (static CSS/JS only).

## Testing

- Ad-hoc scripts: `app/accuracy_test.py`, `app/greedy_vs_n2.py`, `app/RCF_test.py`, `app/tfidf_test.py`, `app/test.py`, `app/newtest.py`.
- No pytest/unittest layout or CI configuration observed. Run scripts manually as needed.

## Common Development Tasks

- **Change port:** Modify `app.run(debug=True, port=5001)` in `server.py` (or use Flask’s env/CLI).
- **Use SQLite for similarity:** Set `USE_DB = True` in `app/recommeder.py` and ensure `MovieLens.db` path is correct; ensure tables exist (e.g. `greedyF_user_similarity_matrix` / `user_similarity_matrix` as in code).
- **Point to different data:** Edit path constants in `app/get_dataset.py` (and any similar paths in `utils.py`).
