# tests/conftest.py  (החלף את החלק העליון בקוד הבא)
import os
import sys
import importlib
import tempfile
import shutil
import json
import pytest
from pathlib import Path

# הוספת שורש הפרויקט ל-sys.path כדי ש-import dal/server יעבדו
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dal as dal_module
from fastapi.testclient import TestClient

@pytest.fixture()  # <-- scope ברירת מחדל: function (לא session)
def temp_dir():
    d = tempfile.mkdtemp(prefix="ml_server_tests_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

@pytest.fixture()
def client(temp_dir, monkeypatch):
    # DB חדש לכל טסט
    test_db_path = os.path.join(temp_dir, "test.db")
    monkeypatch.setattr(dal_module, "DB_NAME", test_db_path, raising=True)
    dal_module.init_db()

    # טען/רענן את האפליקציה אחרי שינוי DB_NAME
    if "server" in list(importlib.sys.modules.keys()):
        importlib.reload(importlib.import_module("server"))
    import server
    importlib.reload(server)
    return TestClient(server.app)

@pytest.fixture()
def signup_and_login(client):
    def _go(username="testuser", password="Aa!1234"):
        r = client.post("/signup", json={"username": username, "password": password})
        assert r.status_code in (200, 409)
        r = client.post("/login", json={"username": username, "password": password})
        assert r.status_code == 200
        token = r.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}, username
    return _go
