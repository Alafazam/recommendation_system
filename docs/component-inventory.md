# Component Inventory: recommendation_system

## Overview

The UI is **server-rendered** (Jinja2). There is no component library or SPA framework; “components” are pages and shared assets.

## Pages (Templates)

| Template | Route(s) | Purpose |
|----------|----------|---------|
| `index.html` | `/` | Home: title “Demo Movie Recommender”, subtitle, “Get Started” button linking to `/movies`. |
| `rating.html` | `/movies`, `/genres/<genre>` | List of movies (with optional limit/order); receives `movies` from backend. |
| `movieDetails.html` | `/movie/<movieId>` | Single movie details; receives `movie` (Movie object). |
| `test.html` | `/test`, `/user` | Test/data view; receives `data` (e.g. movie names or rated movies) and optionally `datax` (user string). |

## Layout and Styling

- **CSS:** Materialize (e.g. `static/stylesheets/materialize.min.css`). Referenced from templates (e.g. `../static/stylesheets/materialize.min.css`).
- **Layout:** Container/row/column structure; teal accent; no separate design system beyond Materialize.

## Static Assets

- **Location:** `app/static/` (e.g. stylesheets, images; 764+ PNGs observed).
- **Usage:** Linked from templates; no bundler or asset pipeline.

## Reusable Elements

- **Navigation:** No shared nav partial observed; each page is self-contained.
- **Movie list:** Rendered in `rating.html` from `movies` list; movie cards/details in `movieDetails.html`.
- **No shared header/footer partials** documented in the scanned templates.

## Backend “Components” (Logical)

- **Recommendation engine:** `recommeder.py` (getRecommendations, similar users), `baseAlgos.py` (similarity matrix, greedy filtering, TF-IDF), `similarityMeasures.py` (cosine, Pearson).
- **Data layer:** `get_dataset.py` (loaders), `utils.py` (Movie, rating helpers). These are not UI components but support the pages above.
