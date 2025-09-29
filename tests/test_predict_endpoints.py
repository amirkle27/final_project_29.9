# tests/test_predict_endpoints.py
import json
import io
import random

def _make_classification_csv():
    # בעיה בינארית פשוטה: size = "S" אם x1+x2 < סף, אחרת "M"
    rows = ["x1,x2,size"]
    for i in range(60):
        x1 = random.randint(0, 20)
        x2 = random.randint(0, 20)
        size = "S" if (x1 + x2) < 20 else "M"
        rows.append(f"{x1},{x2},{size}")
    return "\n".join(rows).encode("utf-8")

def _make_regression_csv():
    rows = ["f1,f2,target"]
    for i in range(40):
        f1 = i
        f2 = 2*i
        target = 5 + 1.5*f1 + 0.5*f2
        rows.append(f"{f1},{f2},{target}")
    return "\n".join(rows).encode("utf-8")

def test_predict_classification(client, signup_and_login):
    headers, _ = signup_and_login()
    csv_bytes = _make_classification_csv()
    files = {"file": ("toy_cls.csv", io.BytesIO(csv_bytes), "text/csv")}
    # מודל ברירת מחדל: logreg (ע"פ השרת)
    r = client.post(
        "/predict/classification",
        headers=headers,
        data={
            "target_col": "size",
            "model": "logreg",
            "data": json.dumps({"x1": 7, "x2": 4})
        },
        files=files
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "prediction" in body
    assert "metrics" in body and "accuracy" in body["metrics"]

def test_predict_regression(client, signup_and_login):
    headers, _ = signup_and_login()
    csv_bytes = _make_regression_csv()
    files = {"file": ("toy_reg.csv", io.BytesIO(csv_bytes), "text/csv")}
    r = client.post(
        "/predict/regression",
        headers=headers,
        data={
            "target_col": "target",
            "model": "linear",
            "data": json.dumps({"f1": 10, "f2": 20})
        },
        files=files
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "prediction" in body
    assert "metrics" in body and {"mse", "rmse", "r2"} <= set(body["metrics"].keys())
