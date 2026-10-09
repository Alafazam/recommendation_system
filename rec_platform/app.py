"""Platform API: data intake, config/masters, recommendations. Domain-agnostic."""
import os
from flask import Flask, request, jsonify

from .db import (
    init_db,
    get_all_items,
    get_items_by_ids,
    get_all_users,
    get_all_ratings,
    get_config,
    set_config,
    upsert_items,
    upsert_users,
    upsert_ratings,
)
from .engine import RecommendationEngine

app = Flask(__name__)
init_db()

engine = RecommendationEngine(get_all_items, get_all_users, get_all_ratings)


# ---------- Data intake ----------
@app.route("/api/v1/items", methods=["GET", "POST"])
def api_items():
    if request.method == "GET":
        ids = request.args.getlist("id", type=int)
        if ids:
            items = get_items_by_ids(ids)
        else:
            items = get_all_items()
        return jsonify({"items": items})
    body = request.get_json(force=True, silent=True) or []
    if isinstance(body, dict):
        body = [body]
    for it in body:
        if "external_id" not in it:
            return jsonify({"error": "external_id required"}), 400
    upsert_items(body)
    return jsonify({"status": "ok", "count": len(body)})


@app.route("/api/v1/users", methods=["GET", "POST"])
def api_users():
    if request.method == "GET":
        return jsonify({"users": get_all_users()})
    body = request.get_json(force=True, silent=True) or []
    if isinstance(body, dict):
        body = [body]
    for u in body:
        if "external_id" not in u:
            return jsonify({"error": "external_id required"}), 400
    upsert_users(body)
    return jsonify({"status": "ok", "count": len(body)})


@app.route("/api/v1/ratings", methods=["GET", "POST"])
def api_ratings():
    if request.method == "GET":
        return jsonify({"ratings": get_all_ratings()})
    body = request.get_json(force=True, silent=True) or []
    if isinstance(body, dict):
        body = [body]
    upsert_ratings(body)
    return jsonify({"status": "ok", "count": len(body)})


# ---------- Config / masters ----------
@app.route("/api/v1/config", methods=["GET", "PUT"])
def api_config():
    if request.method == "GET":
        return jsonify(get_config())
    body = request.get_json(force=True, silent=True) or {}
    for k, v in body.items():
        set_config(k, v)
    if "default_algo" in body:
        engine.set_algo(body["default_algo"])
    return jsonify(get_config())


# ---------- Recommendations ----------
@app.route("/api/v1/recommendations", methods=["GET"])
def api_recommendations():
    user_id = request.args.get("user_id", type=int)
    if user_id is None:
        return jsonify({"error": "user_id required"}), 400
    limit = request.args.get("limit", default=10, type=int)
    algo = request.args.get("algo")
    if algo:
        engine.set_algo(algo)
    recs = engine.get_recommendations_for_user(user_id, limit=limit, algo=algo)
    return jsonify({"recommendations": [{"item_id": iid, "score": float(s)} for iid, s in recs]})


@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
