"""Movie product: HTTP client to platform API. No recommendation logic."""
import os
import requests

PLATFORM_URL = os.environ.get("PLATFORM_URL", "http://localhost:5001")
TIMEOUT = 10


def _get(path, params=None):
    r = requests.get(f"{PLATFORM_URL}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(path, json_body):
    r = requests.post(f"{PLATFORM_URL}{path}", json=json_body, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_all_items():
    data = _get("/api/v1/items")
    return data.get("items") or []


def get_items_by_ids(ids):
    if not ids:
        return []
    data = _get("/api/v1/items", params=[("id", i) for i in ids])
    return data.get("items") or []


def get_all_users():
    data = _get("/api/v1/users")
    return data.get("users") or []


def get_recommendations(user_id, limit=10, algo=None):
    params = {"user_id": user_id, "limit": limit}
    if algo:
        params["algo"] = algo
    data = _get("/api/v1/recommendations", params=params)
    return data.get("recommendations") or []


def item_to_movie(it):
    """Map platform item to movie dict for templates."""
    att = it.get("attributes") or {}
    return {
        "id": it["id"],
        "movieId": it["id"],
        "name": att.get("name") or it.get("external_id", ""),
        "imdbUrl": att.get("imdb_url") or "",
        "genres": att.get("genres") or [],
        "overallRating": att.get("avg_rating") or 0,
    }
