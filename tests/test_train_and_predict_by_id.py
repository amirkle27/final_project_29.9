"""
Integration test for the `/train` and `/predict/by_id/{model_id}` endpoints
of the FastAPI ML Server.

This test ensures that:
- Models can be successfully trained and stored in the database.
- Predictions can be made later using the stored model via its unique model_id.
- The API returns proper status codes and response structures.
"""

import json
import io

def _make_regression_csv():
    """
    Generate a small in-memory CSV dataset for regression testing.

    The dataset contains two features (`x1`, `x2`) and a target variable `y`
    defined by the linear relationship: y = 2*x1 + 3*x2.

    Returns:
        bytes: Encoded CSV file content suitable for upload.
    """
    rows = ["x1,x2,y"]
    for i in range(50):
        x1 = i
        x2 = 2*i
        y = 2*x1 + 3*x2
        rows.append(f"{x1},{x2},{y}")
    return "\n".join(rows).encode("utf-8")

def test_train_and_predict_by_id_flow(client, signup_and_login):
    """
    Test the complete workflow of model training and prediction by model ID.

    Steps:
        1. Log in to obtain an authorization header.
        2. Upload a regression dataset to `/train` to train a linear model.
        3. Verify the model was trained successfully and a model_id was returned.
        4. Send a `/predict/by_id/{model_id}` request using the stored model.
        5. Verify that the response includes a numeric prediction value.

    Verifies:
        - The `/train` endpoint returns 200 and correct JSON fields.
        - The `/predict/by_id` endpoint works correctly using a model_id.
    """
    headers, _ = signup_and_login()

    # 1) model training
    csv_bytes = _make_regression_csv()
    files = {"file": ("toy_reg.csv", io.BytesIO(csv_bytes), "text/csv")}
    data = {
        "model_name": "linear",
        "features": json.dumps(["x1", "x2"]),
        "label": "y",
        "model_params": json.dumps({})  # אופציונלי
    }
    r = client.post("/train", headers=headers, data=data, files=files)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "model trained"
    assert j["model_name"] == "linear"
    model_id = j["model_id"]

    # 2) prediction by model_id
    new_row = {"x1": 10, "x2": 20}
    r = client.post(f"/predict/by_id/{model_id}", headers=headers, data={"data": json.dumps(new_row)})
    assert r.status_code == 200, r.text
    pred = r.json()["prediction"]
    assert isinstance(pred, float) or isinstance(pred, int)


