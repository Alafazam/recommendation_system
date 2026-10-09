# Project Overview: recommendation_system

## Purpose

A **movie recommendation system** using collaborative filtering. The app serves a web UI for browsing movies and genres, viewing movie details, and uses user-based and item-based recommendation algorithms (cosine and Pearson similarity) over the MovieLens 100k dataset.

## Executive Summary

- **Type:** Monolith (single codebase)
- **Stack:** Python, Flask, NumPy, SciPy; server-rendered HTML (Jinja2) with Materialize CSS
- **Data:** MovieLens 100k (ratings, movies, users, genres); in-memory by default with optional SQLite
- **Architecture:** Request/response web app with recommendation engine in-process

## Tech Stack Summary

| Category     | Technology        | Notes                          |
|-------------|--------------------|--------------------------------|
| Runtime     | Python 2.x         | (print statements, xrange)     |
| Web         | Flask              | Routes, templates, debug server|
| Data        | NumPy arrays       | V (ratings), U (users), ALL_MOVIES |
| Optional DB | SQLite             | MovieLens.db, similarity cache |
| ML/Similarity | SciPy, custom    | cosine_similarity, pearson_similarity |
| Frontend    | Jinja2 + Materialize CSS | Server-rendered pages   |

## Repository Structure

- **Monolith:** One application (no separate client/server repos).
- **Entry:** `server.py` runs the Flask app; `run.py` is a standalone script for CLI-style recommendation.
- **Core logic:** `app/` (Flask app, recommender, similarity, dataset loading, templates, static assets).

## Links to Detailed Docs

- [Architecture](./architecture.md)
- [Source Tree Analysis](./source-tree-analysis.md)
- [Development Guide](./development-guide.md)
- [API / Routes](./api-contracts.md)
- [Data Models & Dataset](./data-models.md)
- [Component Inventory](./component-inventory.md)

## Getting Started (Quick)

1. Ensure Python 2 and dependencies (Flask, NumPy, SciPy) are installed.
2. From project root: `python server.py` (or run the Flask app via `app`).
3. Open http://localhost:5000 (or the port Flask prints).
4. Use “Get Started” to browse movies and genres; optional `/user?userid=<id>` for user view.

See [Development Guide](./development-guide.md) for full setup and run instructions.
