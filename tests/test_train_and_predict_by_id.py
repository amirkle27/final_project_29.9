# tests/test_train_and_predict_by_id.py
import json
import io

def _make_regression_csv():
    # דאטה צעצוע לרגרסיה: y = 2*x1 + 3*x2 + רעש קטן
    rows = ["x1,x2,y"]
    for i in range(50):
        x1 = i
        x2 = 2*i
        y = 2*x1 + 3*x2
        rows.append(f"{x1},{x2},{y}")
    return "\n".join(rows).encode("utf-8")

def test_train_and_predict_by_id_flow(client, signup_and_login):
    headers, _ = signup_and_login()

    # 1) אימון
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

    # 2) חיזוי לפי model_id
    new_row = {"x1": 10, "x2": 20}
    r = client.post(f"/predict/by_id/{model_id}", headers=headers, data={"data": json.dumps(new_row)})
    assert r.status_code == 200, r.text
    pred = r.json()["prediction"]
    assert isinstance(pred, float) or isinstance(pred, int)
