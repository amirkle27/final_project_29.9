"""
Integration tests for training and prediction using non-CSV file formats
(Excel and JSON) in the FastAPI ML Server.

These tests verify that:
- Excel (.xlsx) and JSON (.json) files can be uploaded, converted, and processed correctly.
- Models can be trained from both file types.
- Predictions can be made successfully using the trained model's ID.
"""

import io, json
import pandas as pd

def _make_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a pandas DataFrame into in-memory Excel (.xlsx) bytes.

    Args:
        df (pd.DataFrame): The DataFrame to save as an Excel file.

    Returns:
        bytes: Excel file contents encoded in memory, ready for upload.
    """
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return bio.getvalue()

def test_train_from_excel_and_predict(client, signup_and_login):
    """
    Test training and prediction using an Excel (.xlsx) file.

    Steps:
        1. Log in to obtain an auth token.
        2. Create a small regression DataFrame and convert it to Excel bytes.
        3. Upload the Excel file to `/train` with model_name="linear".
        4. Verify successful training (HTTP 200) and extract `model_id`.
        5. Send a `/predict/by_id/{model_id}` request and verify a prediction is returned.
    """
    headers, _ = signup_and_login()

    df = pd.DataFrame({"f1":[0,1,2,3,4,5], "f2":[1,2,3,4,5,6]})
    df["y"] = 3 + 1.2*df["f1"] + 0.8*df["f2"]
    xlsx_bytes = _make_excel_bytes(df)

    files = {"file": ("reg_data.xlsx", io.BytesIO(xlsx_bytes), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    data = {"model_name": "linear", "features": json.dumps(["f1","f2"]), "label":"y"}
    r = client.post("/train", headers=headers, data=data, files=files)
    assert r.status_code == 200, r.text
    model_id = r.json()["model_id"]

    r = client.post(f"/predict/by_id/{model_id}", headers=headers, data={"data": json.dumps({"f1":10, "f2":11})})
    assert r.status_code == 200
    assert "prediction" in r.json()

def test_train_from_json_and_predict(client, signup_and_login):
    """
    Test training and prediction using a JSON (.json) file.

    Steps:
        1. Log in to obtain an auth token.
        2. Generate a small classification dataset in JSON format.
        3. Upload the JSON file to `/train` with model_name="logreg".
        4. Verify training succeeds (HTTP 200) and retrieve `model_id`.
        5. Send a `/predict/by_id/{model_id}` request and confirm prediction output.
    """
    headers, _ = signup_and_login()

    rows = []
    for i in range(30):
        x1 = i % 20
        x2 = (i*2) % 20
        size = "S" if (x1+x2) < 18 else "M"
        rows.append({"x1": x1, "x2": x2, "size": size})

    json_bytes = io.BytesIO(json.dumps(rows).encode("utf-8"))
    files = {"file": ("cls_data.json", json_bytes, "application/json")}
    data = {"model_name": "logreg", "features": json.dumps(["x1","x2"]), "label":"size"}
    r = client.post("/train", headers=headers, data=data, files=files)
    assert r.status_code == 200, r.text
    model_id = r.json()["model_id"]

    r = client.post(f"/predict/by_id/{model_id}", headers=headers, data={"data": json.dumps({"x1": 5, "x2": 9})})
    assert r.status_code == 200
    assert "prediction" in r.json()


