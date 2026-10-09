"""Recommendation engine: in-memory matrix, similarity, recommendations. Domain-agnostic."""
import numpy as np


def cosine_similarity(v1, v2):
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    nz = np.logical_or(v1 != 0, v2 != 0)
    if not np.any(nz):
        return 0.0
    a, b = v1[nz], v2[nz]
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def pearson_similarity(r1, r2):
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    idx = np.intersect1d(np.nonzero(r1)[0], np.nonzero(r2)[0])
    if len(idx) == 0:
        return 0.0
    x, y = r1[idx], r2[idx]
    n = len(x)
    sum_xy = np.sum(x * y)
    sum_x, sum_y = np.sum(x), np.sum(y)
    sum_x2, sum_y2 = np.sum(x ** 2), np.sum(y ** 2)
    den = np.sqrt(sum_x2 - (sum_x ** 2) / n) * np.sqrt(sum_y2 - (sum_y ** 2) / n)
    if den == 0:
        return 0.0
    return (sum_xy - (sum_x * sum_y) / n) / den


def build_user_similarity_matrix(V, similarity_fn=cosine_similarity, min_sim=0.0, top_k=50):
    """V: (n_users, n_items). Returns (n_users, n_users) similarity matrix."""
    n = V.shape[0]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = similarity_fn(V[i], V[j])
            if s > min_sim:
                sim[i, j] = sim[j, i] = s
    return sim


def get_top_similar_users(sim_matrix, user_idx, k=20):
    row = sim_matrix[user_idx]
    candidates = [(row[j], j) for j in range(len(row)) if row[j] > 0 and j != user_idx]
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[:k]


def get_recommendations(V, user_idx, similar_users, limit=10):
    """V (n_users, n_items); similar_users list of (sim, other_user_idx). Return list of (score, item_idx)."""
    n_items = V.shape[1]
    u_rated = set(np.nonzero(V[user_idx])[0].tolist())
    V_item_user = V.T  # (n_items, n_users)
    scores = []
    for item_idx in range(n_items):
        if item_idx in u_rated:
            continue
        score_sum, sim_sum = 0.0, 0.0
        for sim, other_idx in similar_users:
            r = V_item_user[item_idx, other_idx]
            if r > 0:
                score_sum += r * sim
                sim_sum += sim
        if sim_sum > 0:
            scores.append((score_sum / sim_sum, item_idx))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:limit]


class RecommendationEngine:
    """In-memory engine: loads from DB layer (items, users, ratings), exposes get_recommendations."""

    def __init__(self, get_all_items_fn, get_all_users_fn, get_all_ratings_fn):
        self._get_items = get_all_items_fn
        self._get_users = get_all_users_fn
        self._get_ratings = get_all_ratings_fn
        self._V = None
        self._user_id_to_idx = None
        self._item_id_to_idx = None
        self._idx_to_item_id = None
        self._sim_matrix = None
        self._algo = "cosine"

    def _build_matrix(self):
        items = self._get_items()
        users = self._get_users()
        ratings = self._get_ratings()
        if not items or not users:
            self._V = np.zeros((0, 0))
            self._user_id_to_idx = {}
            self._item_id_to_idx = {}
            self._idx_to_item_id = []
            return
        self._user_id_to_idx = {u["id"]: i for i, u in enumerate(users)}
        self._item_id_to_idx = {it["id"]: j for j, it in enumerate(items)}
        self._idx_to_item_id = [it["id"] for it in items]
        n_u, n_i = len(users), len(items)
        V = np.zeros((n_u, n_i))
        for r in ratings:
            ui = self._user_id_to_idx.get(r["user_id"])
            ii = self._item_id_to_idx.get(r["item_id"])
            if ui is not None and ii is not None:
                V[ui, ii] = r["rating"]
        self._V = V
        self._sim_matrix = None

    def _ensure_loaded(self):
        if self._V is None:
            self._build_matrix()

    def set_algo(self, algo):
        self._algo = algo if algo in ("cosine", "pearson") else "cosine"

    def get_recommendations_for_user(self, user_id, limit=10, algo=None):
        self._ensure_loaded()
        if self._V.size == 0:
            return []
        user_idx = self._user_id_to_idx.get(user_id)
        if user_idx is None:
            return []
        use_algo = algo or self._algo
        sim_fn = cosine_similarity if use_algo == "cosine" else pearson_similarity
        if self._sim_matrix is None or use_algo != getattr(self, "_last_algo", None):
            self._sim_matrix = build_user_similarity_matrix(self._V, similarity_fn=sim_fn)
            self._last_algo = use_algo
        similar = get_top_similar_users(self._sim_matrix, user_idx, k=30)
        recs = get_recommendations(self._V, user_idx, similar, limit=limit)
        return [(self._idx_to_item_id[idx], score) for score, idx in recs]

    def get_item_ids_ordered(self):
        self._ensure_loaded()
        return list(self._idx_to_item_id) if self._idx_to_item_id else []
