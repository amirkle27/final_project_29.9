def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

def test_signup_login_flow(client):
    r = client.post("/signup", json={"username": "alice", "password": "Ab!1234"})
    assert r.status_code in (200, 409)
    r = client.post("/login", json={"username": "alice", "password": "Ab!1234"})
    assert r.status_code == 200
    assert "access_token" in r.json()

