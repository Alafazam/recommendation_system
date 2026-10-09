"""Platform DB: domain-agnostic schema (items, users, ratings, config)."""
import os
import sqlite3
import json

_basedir = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.environ.get("PLATFORM_DB", os.path.join(_basedir, "rec_platform.db"))


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT UNIQUE NOT NULL,
            attributes TEXT DEFAULT '{}',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT UNIQUE NOT NULL,
            attributes TEXT DEFAULT '{}',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS ratings (
            user_id INTEGER NOT NULL REFERENCES users(id),
            item_id INTEGER NOT NULL REFERENCES items(id),
            rating REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, item_id)
        );
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT OR IGNORE INTO config (key, value) VALUES ('default_algo', 'cosine');
    """)
    conn.commit()
    conn.close()


def upsert_items(items):
    """items: list of {external_id, attributes}."""
    conn = get_conn()
    for it in items:
        attrs = json.dumps(it.get("attributes") or {})
        try:
            conn.execute("INSERT INTO items (external_id, attributes) VALUES (?, ?)", (it["external_id"], attrs))
        except sqlite3.IntegrityError:
            conn.execute("UPDATE items SET attributes = ? WHERE external_id = ?", (attrs, it["external_id"]))
    conn.commit()
    conn.close()


def upsert_users(users):
    conn = get_conn()
    for u in users:
        attrs = json.dumps(u.get("attributes") or {})
        try:
            conn.execute("INSERT INTO users (external_id, attributes) VALUES (?, ?)", (u["external_id"], attrs))
        except sqlite3.IntegrityError:
            conn.execute("UPDATE users SET attributes = ? WHERE external_id = ?", (attrs, u["external_id"]))
    conn.commit()
    conn.close()


def upsert_ratings(ratings):
    """ratings: list of {user_external_id, item_external_id, rating} or {user_id, item_id, rating}."""
    conn = get_conn()
    cur = conn.cursor()
    for r in ratings:
        if "user_id" in r and "item_id" in r:
            uid, iid = r["user_id"], r["item_id"]
        else:
            cur.execute("SELECT id FROM users WHERE external_id = ?", (r["user_external_id"],))
            row = cur.fetchone()
            if not row:
                continue
            uid = row[0]
            cur.execute("SELECT id FROM items WHERE external_id = ?", (r["item_external_id"],))
            row = cur.fetchone()
            if not row:
                continue
            iid = row[0]
        cur.execute(
            "INSERT OR REPLACE INTO ratings (user_id, item_id, rating) VALUES (?, ?, ?)",
            (uid, iid, float(r["rating"])),
        )
    conn.commit()
    conn.close()


def get_all_items():
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, external_id, attributes FROM items ORDER BY id").fetchall()
    conn.close()
    return [{"id": r[0], "external_id": r[1], "attributes": json.loads(r[2] or "{}")} for r in rows]


def get_items_by_ids(ids):
    if not ids:
        return []
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, external_id, attributes FROM items WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    conn.close()
    return [{"id": r[0], "external_id": r[1], "attributes": json.loads(r[2] or "{}")} for r in rows]


def get_all_users():
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, external_id, attributes FROM users ORDER BY id").fetchall()
    conn.close()
    return [{"id": r[0], "external_id": r[1], "attributes": json.loads(r[2] or "{}")} for r in rows]


def get_all_ratings():
    conn = get_conn()
    rows = conn.execute("SELECT user_id, item_id, rating FROM ratings").fetchall()
    conn.close()
    return [{"user_id": r[0], "item_id": r[1], "rating": r[2]} for r in rows]


def get_config():
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM config").fetchall()
    conn.close()
    return dict(rows)


def set_config(key, value):
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()
