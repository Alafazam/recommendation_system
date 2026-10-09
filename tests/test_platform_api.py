"""Platform API tests: health, config, data intake, recommendations."""
import pytest


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_config_get(client):
    r = client.get("/api/v1/config")
    assert r.status_code == 200
    data = r.get_json()
    assert "default_algo" in data


def test_config_put(client):
    r = client.put("/api/v1/config", json={"default_algo": "pearson"})
    assert r.status_code == 200
    assert r.get_json().get("default_algo") == "pearson"
    r = client.put("/api/v1/config", json={"default_algo": "cosine"})
    assert r.status_code == 200


def test_items_post_and_get(client):
    r = client.post("/api/v1/items", json=[
        {"external_id": "i1", "attributes": {"name": "Item 1"}},
        {"external_id": "i2", "attributes": {"name": "Item 2"}},
    ])
    assert r.status_code == 200
    assert r.get_json()["count"] == 2

    r = client.get("/api/v1/items")
    assert r.status_code == 200
    items = r.get_json()["items"]
    assert len(items) >= 2
    ids = [x["id"] for x in items if x["external_id"] in ("i1", "i2")]
    assert len(ids) == 2


def test_items_post_single(client):
    r = client.post("/api/v1/items", json={"external_id": "i_single", "attributes": {}})
    assert r.status_code == 200
    assert r.get_json()["count"] == 1


def test_items_post_missing_external_id(client):
    r = client.post("/api/v1/items", json=[{"attributes": {}}])
    assert r.status_code == 400


def test_users_post_and_get(client):
    r = client.post("/api/v1/users", json=[
        {"external_id": "u1", "attributes": {}},
        {"external_id": "u2", "attributes": {}},
    ])
    assert r.status_code == 200
    assert r.get_json()["count"] == 2

    r = client.get("/api/v1/users")
    assert r.status_code == 200
    users = r.get_json()["users"]
    assert len(users) >= 2


def test_ratings_post_with_external_ids(client):
    # Ensure items and users exist
    client.post("/api/v1/items", json=[
        {"external_id": "r_i1", "attributes": {}},
        {"external_id": "r_i2", "attributes": {}},
    ])
    client.post("/api/v1/users", json=[
        {"external_id": "r_u1", "attributes": {}},
        {"external_id": "r_u2", "attributes": {}},
    ])
    r = client.post("/api/v1/ratings", json=[
        {"user_external_id": "r_u1", "item_external_id": "r_i1", "rating": 4.0},
        {"user_external_id": "r_u1", "item_external_id": "r_i2", "rating": 3.0},
        {"user_external_id": "r_u2", "item_external_id": "r_i1", "rating": 5.0},
    ])
    assert r.status_code == 200
    assert r.get_json()["count"] == 3


def test_recommendations_requires_user_id(client):
    r = client.get("/api/v1/recommendations")
    assert r.status_code == 400
    assert "user_id" in r.get_json().get("error", "").lower()


def test_recommendations_returns_list(client):
    # With no data, may return []
    client.post("/api/v1/users", json=[{"external_id": "rec_u", "attributes": {}}])
    users = client.get("/api/v1/users").get_json()["users"]
    uid = next(u["id"] for u in users if u["external_id"] == "rec_u")
    r = client.get("/api/v1/recommendations", query_string={"user_id": uid, "limit": 5})
    assert r.status_code == 200
    data = r.get_json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
