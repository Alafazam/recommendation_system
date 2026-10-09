"""Seed platform DB from MovieLens 100k files. Run from repo root: python scripts/seed_platform_from_ml100k.py"""
import os
import sys
import json

# Repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from rec_platform.db import init_db, upsert_items, upsert_users, upsert_ratings, get_conn

ML100K = os.path.join(ROOT, "app", "ml-100k")
if not os.path.isdir(ML100K):
    ML100K = os.path.join(ROOT, "ml-100k")
if not os.path.isdir(ML100K):
    print("ml-100k dir not found under app/ or repo root")
    sys.exit(1)

def load_genres(path):
    genres = []
    with open(os.path.join(path, "u.genre")) as f:
        for line in f:
            part = line.strip().split("|")
            if part[0]:
                genres.append(part[0])
    return genres

def main():
    init_db()
    genres = load_genres(ML100K)

    # Items (movies): external_id = movie id from u.item, attributes = name, imdb_url, genre list
    items = []
    with open(os.path.join(ML100K, "u.item"), encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if not parts[0]:
                continue
            movie_id = parts[0].strip()
            name = (parts[1] or "").strip()
            imdb_url = (parts[4] or "").strip()
            genre_vec = [int(x) for x in (parts[5:5+19] if len(parts) >= 24 else [])]
            genre_names = [genres[i] for i in range(len(genre_vec)) if genre_vec[i] > 0]
            items.append({
                "external_id": f"m{movie_id}",
                "attributes": {"name": name, "imdb_url": imdb_url, "genres": genre_names, "movie_id": movie_id},
            })
    upsert_items(items)
    print(f"Upserted {len(items)} items (movies)")

    # Users
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
    upsert_users(users)
    print(f"Upserted {len(users)} users")

    # Ratings: need internal item/user ids. After upsert, DB has id 1..n for items and users by insert order.
    # u.item is 1-indexed movie id; u.user is 1-indexed user id. u.data: user_id, movie_id, rating, ts
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, external_id FROM items")
    ext_to_item_id = {r[1]: r[0] for r in cur.fetchall()}
    cur.execute("SELECT id, external_id FROM users")
    ext_to_user_id = {r[1]: r[0] for r in cur.fetchall()}
    conn.close()

    ratings = []
    with open(os.path.join(ML100K, "u.data")) as f:
        for line in f:
            u_id, m_id, rating, _ = line.strip().split()
            ext_u, ext_m = f"u{u_id}", f"m{m_id}"
            if ext_u in ext_to_user_id and ext_m in ext_to_item_id:
                ratings.append({
                    "user_id": ext_to_user_id[ext_u],
                    "item_id": ext_to_item_id[ext_m],
                    "rating": float(rating),
                })
    # upsert_ratings expects user_id/item_id as internal ids
    upsert_ratings(ratings)
    print(f"Upserted {len(ratings)} ratings")

if __name__ == "__main__":
    main()
