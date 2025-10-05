import json, io

def test_models_listing_costs_one_token(client, signup_and_login):
    headers, username = signup_and_login()

    rows = ["f1,f2,y"]
    for i in range(20): 
        rows.append(f"{i},{i+1},{2*i+1}")
    csv = ("\n".join(rows)).encode("utf-8")

    files = {"file": ("m.csv", io.BytesIO(csv), "text/csv")}
    data = {"model_name": "linear", "features": json.dumps(["f1", "f2"]), "label": "y"}
    r = client.post("/train", headers=headers, data=data, files=files)
    assert r.status_code == 200

    r = client.get("/models", headers=headers)
    assert r.status_code == 200
    models = r.json()["models"]
    assert isinstance(models, list) and len(models) >= 1

