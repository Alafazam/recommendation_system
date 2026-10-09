"""Pytest fixtures: platform app test client and API base URL."""
import os
import pytest

# Use in-memory SQLite for tests so we don't touch the real DB
os.environ["PLATFORM_DB"] = ":memory:"


@pytest.fixture
def client():
    """Flask test client for the platform API."""
    from rec_platform.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def api_base():
    """Base URL for API (used by load script tests)."""
    return "http://localhost:5001"
