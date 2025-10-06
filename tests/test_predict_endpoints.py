"""
Integration tests for the `/predict/classification` and `/predict/regression` endpoints
of the FastAPI ML Server.

These tests ensure that:
- Classification models can be trained and used to make predictions.
- Regression models work correctly and return proper evaluation metrics.
- Both endpoints return valid JSON responses containing predictions and metrics.
"""

import json
import io
import random

def _make_classification_csv():
    """
    Generate an in-memory CSV dataset for classification testing.

    The dataset has two numeric features (`x1`, `x2`) and one categorical label (`size`),
    where the label is 'S' if x1 + x2 < 20, otherwise 'M'.

    Returns:
        bytes: Encoded CSV data suitable for uploading via `io.BytesIO`.
    """
    rows = ["x1,x2,size"]
    for i in range(60):
        x1 = random.randint(0, 20)
        x2 = random.randint(0, 20)
        size = "S" if (x1 + x2) < 20 else "M"
        rows.append(f"{x1},{x2},{size}")
    return "\n".join(rows).encode("utf-8")

def _make_regression_csv():
    """
    Generate an in-memory CSV dataset for regression testing.

    The dataset has two numeric features (`f1`, `f2`) and a continuous target (`target`),
    computed as: target = 5 + 1.5*f1 + 0.5*f2.

    Returns:
        bytes: Encoded CSV data suitable for uploading via `io.BytesIO`.
    """
    rows = ["f1,f2,target"]
    for i in range(40):
        f1 = i
        f2 = 2*i
        target = 5 + 1.5*f1 + 0.5*f2
        rows.append(f"{f1},{f2},{target}")
    return "\n".join(rows).encode("utf-8")

def test_predict_classification(client, signup_and_login):
    """
    Test the `/predict/classification` endpoint.

    Steps:
        1. Log in to obtain an auth token.
        2. Generate a synthetic classification dataset.
        3. Send a POST request to `/predict/classification` with:
            - Model = "logreg"
            - Target column = "size"
            - Features = {"x1": 7, "x2": 4}
        4. Verify:
            - HTTP 200 status code.
            - JSON response contains "prediction".
            - JSON response includes "metrics" with an "accuracy" key.
    """
    headers, _ = signup_and_login()
    csv_bytes = _make_classification_csv()
    files = {"file": ("toy_cls.csv", io.BytesIO(csv_bytes), "text/csv")}
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
    """
    Test the `/predict/regression` endpoint.

    Steps:
        1. Log in to obtain an auth token.
        2. Generate a synthetic regression dataset.
        3. Send a POST request to `/predict/regression` with:
            - Model = "linear"
            - Target column = "target"
            - Features = {"f1": 10, "f2": 20}
        4. Verify:
            - HTTP 200 status code.
            - JSON response contains "prediction".
            - JSON response includes "metrics" with keys: "mse", "rmse", and "r2".
    """
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


