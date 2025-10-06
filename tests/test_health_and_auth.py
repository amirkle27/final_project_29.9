"""
Basic integration tests for FastAPI ML Server authentication and health endpoints.

This module verifies that:
- The server health check endpoint responds correctly.
- User signup and login flow works as expected.
"""

def test_health(client):
    """
    Test the /health endpoint to ensure the server is running.

    Steps:
        1. Send a GET request to /health.
        2. Verify that the response status code is 200.
        3. Verify that the response JSON matches {"ok": True}.
    """
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

def test_signup_login_flow(client):
    """
    Test the user signup and login workflow.

    Steps:
        1. Attempt to sign up a new user (or re-signup if already exists).
        2. Log in with the same credentials.
        3. Ensure login succeeds and an access token is returned.
    """
    r = client.post("/signup", json={"username": "alice", "password": "Ab!1234"})
    assert r.status_code in (200, 409)
    r = client.post("/login", json={"username": "alice", "password": "Ab!1234"})
    assert r.status_code == 200
    assert "access_token" in r.json()


