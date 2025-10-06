"""
Pytest fixture setup for FastAPI ML Server tests.

This module provides reusable pytest fixtures for:
- Creating a temporary directory and database for each test.
- Initializing a fresh FastAPI TestClient with isolated state.
- Signing up and logging in a test user to retrieve an auth token.

Each test runs in isolation with a unique temporary database, ensuring
clean and predictable results.
"""

import os
import sys
import importlib
import tempfile
import shutil
import json
import pytest
from pathlib import Path
import dal as dal_module
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture()  
def temp_dir():
    """
    Creates a temporary directory for test files.

    Yields:
        str: Path to a temporary directory created for the test.

    Cleans up:
        The directory and its contents are automatically deleted after the test.
    """
    d = tempfile.mkdtemp(prefix="ml_server_tests_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

@pytest.fixture()
def client(temp_dir, monkeypatch):
    """
    Initializes a FastAPI TestClient with a fresh temporary database.

    Steps:
        1. Creates a new SQLite database file inside the temporary directory.
        2. Monkeypatches `dal.DB_NAME` to point to this database.
        3. Initializes the database schema using `dal.init_db()`.
        4. Reloads the FastAPI `server` module to ensure it uses the test DB.

    Args:
        temp_dir (str): Temporary directory path from the `temp_dir` fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for modifying attributes.

    Returns:
        TestClient: A FastAPI TestClient instance connected to the temporary DB.
    """
    test_db_path = os.path.join(temp_dir, "test.db")
    monkeypatch.setattr(dal_module, "DB_NAME", test_db_path, raising=True)
    dal_module.init_db()

    if "server" in list(importlib.sys.modules.keys()):
        importlib.reload(importlib.import_module("server"))
    import server
    importlib.reload(server)
    return TestClient(server.app)

@pytest.fixture()
def signup_and_login(client):
    """
    Helper fixture for registering and logging in a test user.

    Creates a user account (or reuses an existing one if already created),
    logs in, and returns an authorization header containing the JWT token.

    Args:
        client (TestClient): FastAPI TestClient instance.

    Returns:
        function: A callable that performs signup + login and returns:
            tuple:
                - dict: Authorization header (e.g., {"Authorization": "Bearer <token>"})
                - str: The username used for the login
    """
    def _go(username="testuser", password="Aa!1234"):
        r = client.post("/signup", json={"username": username, "password": password})
        assert r.status_code in (200, 409)
        r = client.post("/login", json={"username": username, "password": password})
        assert r.status_code == 200
        token = r.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}, username
    return _go


