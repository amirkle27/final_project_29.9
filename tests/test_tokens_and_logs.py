"""
Integration tests for token deduction and usage logging in the FastAPI ML Server.

These tests verify that:
- Tokens are properly deducted after training and prediction actions.
- Usage logs record each action with updated token balances.
- The system correctly blocks actions when a user has insufficient tokens.
"""

import json
import io
import sqlite3
import dal

def _make_reg_csv():
    """
    Create a simple in-memory CSV file for regression testing.

    The dataset contains two numeric features (`a`, `b`) and a target (`y = a + 2b`).

    Returns:
        bytes: CSV data encoded in UTF-8 for upload.
    """
    rows = ["a,b,y"]
    for i in range(20):
        rows.append(f"{i},{2*i},{i + 2*i}")
    return "\n".join(rows).encode("utf-8")

def _get_tokens(username):
    """
    Retrieve the current token balance for a given user.

    Args:
        username (str): The username to check.

    Returns:
        int: Number of tokens currently available for the user.
    """
    u = dal.get_user(username)
    return int(u["tokens"])

def _get_usage_rows(limit=999):
    """
    Fetch usage log rows directly from the database.

    Args:
        limit (int): Maximum number of rows to return (default 999).

    Returns:
        list[tuple]: Each tuple contains (action, tokens_after_usage).
    """
    with dal._connect() as conn:
        cur = conn.cursor()
        return cur.execute(
            "SELECT action, tokens_after_usage FROM usage_logs ORDER BY id ASC LIMIT ?",
            (limit,)
        ).fetchall()

def test_token_deduction_and_logs(client, signup_and_login):
    """
    Verify token deduction and usage log recording for training and prediction actions.

    Steps:
        1. Log in and record initial token balance.
        2. Train a model (cost = 1 token) and confirm deduction.
        3. Make a prediction using `/predict/by_id` (cost = 5 tokens) and confirm deduction.
        4. Query usage logs and verify both 'train' and 'predict/by_id' actions appear.
    """
    headers, username = signup_and_login()
    start_tokens = _get_tokens(username)

    # practice (1 token)
    csv_bytes = _make_reg_csv()
    files = {"file": ("r.csv", io.BytesIO(csv_bytes), "text/csv")}
    data = {"model_name": "linear", "features": json.dumps(["a", "b"]), "label": "y"}
    r = client.post("/train", headers=headers, data=data, files=files)
    assert r.status_code == 200, r.text
    after_train_tokens = _get_tokens(username)
    assert after_train_tokens == start_tokens - 1

    # predict/by_id (5 tokens)
    model_id = r.json()["model_id"]
    r = client.post(f"/predict/by_id/{model_id}", headers=headers, data={"data": json.dumps({"a": 10, "b": 20})})
    assert r.status_code == 200
    after_predict_tokens = _get_tokens(username)
    assert after_predict_tokens == after_train_tokens - 5

    #logs test
    rows = _get_usage_rows()
    actions = [row[0] for row in rows]
    assert "train" in actions
    assert "predict/by_id" in actions

def test_not_enough_tokens_402(client, signup_and_login, monkeypatch):
    """
    Verify behavior when a user runs out of tokens.

    Steps:
        1. Log in as a test user.
        2. Set the user's token balance to 0.
        3. Attempt to train a model.
        4. Confirm the server returns HTTP 402 (Payment Required) or 400 (Bad Request).

    This ensures users cannot perform operations without sufficient tokens.
    """
    headers, username = signup_and_login()

    dal.update_tokens(username, 0)
    csv = "x,y\n1,2\n2,3\n".encode("utf-8")
    files = {"file": ("tiny.csv", io.BytesIO(csv), "text/csv")}
    data = {"model_name": "linear", "features": json.dumps(["x"]), "label": "y"}
    r = client.post("/train", headers=headers, data=data, files=files)
    assert r.status_code == 402 or r.status_code == 400, r.text


