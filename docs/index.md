# Project Documentation Index

## Project Overview

- **Type:** Monolith (single codebase)
- **Primary Language:** Python 2.x
- **Architecture:** Flask server-rendered web app with in-process recommendation engine

## Quick Reference

- **Tech Stack:** Flask, NumPy, SciPy; Jinja2 + Materialize CSS; optional SQLite
- **Entry Point:** `server.py` (Flask), or `run.py` / `mlens.py` for CLI
- **Architecture Pattern:** Request/response monolith; optional DB for similarity cache

## Generated Documentation

- [Project Overview](./project-overview.md)
- [Architecture](./architecture.md)
- [Source Tree Analysis](./source-tree-analysis.md)
- [Component Inventory](./component-inventory.md)
- [Development Guide](./development-guide.md)
- [API Contracts (Routes)](./api-contracts.md)
- [Data Models](./data-models.md)

## Existing Documentation

- [README](../README.md) — Project title (recommendation_system)

## Getting Started

1. Install Python 2 and dependencies: `pip install Flask numpy scipy`
2. Ensure MovieLens 100k data is in `app/ml-100k/` (u.data, u.item, u.user, u.genre, u.occupation)
3. Run: `python server.py`
4. Open http://localhost:5000 and use “Get Started” to browse movies

For full setup, see [Development Guide](./development-guide.md).
