# tests/test_tokens_and_logs.py
import json
import io
import sqlite3
import dal

def _make_reg_csv():
    rows = ["a,b,y"]
    for i in range(20):
        rows.append(f"{i},{2*i},{i + 2*i}")
    return "\n".join(rows).encode("utf-8")

def _get_tokens(username):
    u = dal.get_user(username)
    return int(u["tokens"])

def _get_usage_rows(limit=999):
    with dal._connect() as conn:
        cur = conn.cursor()
        return cur.execute(
            "SELECT action, tokens_after_usage FROM usage_logs ORDER BY id ASC LIMIT ?",
            (limit,)
        ).fetchall()

def test_token_deduction_and_logs(client, signup_and_login):
    headers, username = signup_and_login()
    start_tokens = _get_tokens(username)

    # אימון (אמור לעלות 1 טוקן)
    csv_bytes = _make_reg_csv()
    files = {"file": ("r.csv", io.BytesIO(csv_bytes), "text/csv")}
    data = {"model_name": "linear", "features": json.dumps(["a", "b"]), "label": "y"}
    r = client.post("/train", headers=headers, data=data, files=files)
    assert r.status_code == 200, r.text
    after_train_tokens = _get_tokens(username)
    assert after_train_tokens == start_tokens - 1

    # predict/by_id (אמור לעלות 5 טוקנים)
    model_id = r.json()["model_id"]
    r = client.post(f"/predict/by_id/{model_id}", headers=headers, data={"data": json.dumps({"a": 10, "b": 20})})
    assert r.status_code == 200
    after_predict_tokens = _get_tokens(username)
    assert after_predict_tokens == after_train_tokens - 5

    # בדיקת לוגים נרשמו
    rows = _get_usage_rows()
    actions = [row[0] for row in rows]
    assert "train" in actions
    assert "predict/by_id" in actions

def test_not_enough_tokens_402(client, signup_and_login, monkeypatch):
    headers, username = signup_and_login()

    # נרוקן ידנית טוקנים כדי לאלץ 402
    dal.update_tokens(username, 0)
    # ניסיון אימון — צריך להיכשל עם 402 "not enough tokens" מהשרת (שממפה את ValueError)
    csv = "x,y\n1,2\n2,3\n".encode("utf-8")
    files = {"file": ("tiny.csv", io.BytesIO(csv), "text/csv")}
    data = {"model_name": "linear", "features": json.dumps(["x"]), "label": "y"}
    r = client.post("/train", headers=headers, data=data, files=files)
    # שים לב: בקוד השרת, על חסר טוקנים נזרקת 402 (Payment Required) בתוך try/except
    assert r.status_code == 402 or r.status_code == 400, r.text
