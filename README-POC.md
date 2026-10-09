# Platform + Product POC

Demo: **recommendation platform** (API, data, config, algos) + **movie product** (UI only, calls platform).

## Quick start (Python 3)

```bash
# 1. Install deps
pip install -r requirements-platform.txt

# 2. Start platform (port 5001)
python run_platform.py

# 3. In another terminal: load base data via API only (uses app/ml-100k/)
export PLATFORM_URL=http://localhost:5001   # optional, default localhost:5001
python scripts/load_base_data_via_api.py

# 4. Start product (port 5000)
python run_product.py

# 5. Open http://localhost:5000 — Get Started → movies, user recommendations
```

**Alternative:** Seed the DB directly (no platform running) with `python scripts/seed_platform_from_ml100k.py`, then start platform and product.

## What’s in this POC

- **Platform** (`platform/`, `run_platform.py`)
  - **Data intake:** `POST /api/v1/items`, `POST /api/v1/users`, `POST /api/v1/ratings`
  - **Config:** `GET/PUT /api/v1/config` (e.g. `default_algo`: cosine | pearson)
  - **Recommendations:** `GET /api/v1/recommendations?user_id=1&limit=10&algo=cosine`
  - Own SQLite DB (`rec_platform/rec_platform.db` or `PLATFORM_DB`), domain-agnostic schema (items, users, ratings, config)
  - Algo logic in-platform (cosine / Pearson user-user collaborative filtering)

- **Movie product** (`product/`, `run_product.py`)
  - UI: home, movies list, genres, movie detail, user recommendations
  - Calls platform only (no local recommendation engine)

## Tests

```bash
pip install -r requirements-platform.txt
python -m pytest tests/ -v
```

Tests use an in-memory DB (`PLATFORM_DB=:memory:` in `tests/conftest.py`).

## Env (optional)

- `PLATFORM_URL` — product and load script talk to this (default `http://localhost:5001`)
- `PLATFORM_DB` — platform DB path (default `rec_platform/rec_platform.db`)
- `PORT` — platform default 5001, product default 5000
