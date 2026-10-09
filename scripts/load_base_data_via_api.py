#!/usr/bin/env python3
"""
Load MovieLens 100k base data using only the platform's public APIs.
Requires the platform to be running (e.g. python run_platform.py).
Usage:
  export PLATFORM_URL=http://localhost:5001   # optional
  python scripts/load_base_data_via_api.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML100K = os.path.join(ROOT, "app", "ml-100k")
if not os.path.isdir(ML100K):
    ML100K = os.path.join(ROOT, "ml-100k")
if not os.path.isdir(ML100K):
    print("ml-100k dir not found under app/ or repo root")
    sys.exit(1)

import requests

PLATFORM_URL = os.environ.get("PLATFORM_URL", "http://localhost:5001")
BATCH = 500


def api_get(path, params=None):
    r = requests.get(f"{PLATFORM_URL}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(path, json_body):
    r = requests.post(f"{PLATFORM_URL}{path}", json=json_body, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    print("Checking platform at", PLATFORM_URL)
    api_get("/api/v1/health")
    print("Loading genres...")
    genres = []
    with open(os.path.join(ML100K, "u.genre")) as f:
        for line in f:
            part = line.strip().split("|")
            if part[0]:
                genres.append(part[0])

    # Items (movies) via API
    items = []
    with open(os.path.join(ML100K, "u.item"), encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if not parts[0]:
                continue
            movie_id = parts[0].strip()
            name = (parts[1] or "").strip()
            imdb_url = (parts[4] or "").strip()
            genre_vec = [int(x) for x in (parts[5 : 5 + 19] if len(parts) >= 24 else [])]
            genre_names = [genres[i] for i in range(len(genre_vec)) if genre_vec[i] > 0]
            items.append({
                "external_id": f"m{movie_id}",
                "attributes": {"name": name, "imdb_url": imdb_url, "genres": genre_names, "movie_id": movie_id},
            })

    for i in range(0, len(items), BATCH):
        batch = items[i : i + BATCH]
        api_post("/api/v1/items", batch)
        print(f"  items: {i + len(batch)}/{len(items)}")
    print(f"Posted {len(items)} items")

    # Users via API
    users = []
    with open(os.path.join(ML100K, "u.user")) as f:
        for line in f:
            parts = line.strip().split("|")
            if not parts[0]:
                continue
            uid = parts[0].strip()
            users.append({
                "external_id": f"u{uid}",
                "attributes": {"age": parts[1], "gender": parts[2], "occupation": parts[3], "zip": parts[4]},
            })

    for i in range(0, len(users), BATCH):
        batch = users[i : i + BATCH]
        api_post("/api/v1/users", batch)
        print(f"  users: {i + len(batch)}/{len(users)}")
    print(f"Posted {len(users)} users")

    # Ratings via API (platform accepts user_external_id, item_external_id, rating)
    ratings = []
    with open(os.path.join(ML100K, "u.data")) as f:
        for line in f:
            u_id, m_id, rating, _ = line.strip().split()
            ratings.append({
                "user_external_id": f"u{u_id}",
                "item_external_id": f"m{m_id}",
                "rating": float(rating),
            })

    for i in range(0, len(ratings), BATCH):
        batch = ratings[i : i + BATCH]
        api_post("/api/v1/ratings", batch)
        print(f"  ratings: {i + len(batch)}/{len(ratings)}")
    print(f"Posted {len(ratings)} ratings")
    print("Done.")


if __name__ == "__main__":
    main()
