# API Contracts (Flask Routes)

The application exposes **HTTP routes that return HTML** (server-rendered). There is no separate JSON API.

## Base URL

- Local dev: `http://localhost:5000/` (or the port shown when running `python server.py`)

## Routes

| Method | Path | Query / Path Args | Description |
|--------|------|-------------------|-------------|
| GET | `/` | — | Home page: “Demo Movie Recommender” and “Get Started” link. |
| GET | `/movies` | `limit` (default 100), `order` (1 = desc) | Paginated/sorted list of all movies; renders `rating.html`. |
| GET | `/genres/<genre>` | `limit`, `order` (same as above) | Movies in the given genre; redirects to `/` if genre unknown. Renders `rating.html`. |
| GET | `/movie/<movieId>` | `movieId` (int) | Movie details page; redirects to `/` if id out of range. Renders `movieDetails.html`. |
| GET | `/test` | — | Test view; passes `MovieNames` to `test.html`. |
| GET | `/user` | `userid` (default 552) | User view: rated movies and user info; renders `test.html` with `_names_of_rated` and `user_string`. |

## Request/Response

- **Request:** Standard browser GET; no documented auth or custom headers.
- **Response:** HTML (Jinja2-rendered templates). No JSON endpoints.

## Implementation Reference

- All routes are defined in `app/__init__.py`.
- Data comes from in-memory structures: `ALL_MOVIES`, `Movie_generes`, `Movie_by_generes`, `getUser()`, etc., loaded via `app/get_dataset.py` and `app/utils.py`.
