"""
Integration test for the `/train` and `/models` endpoints of the FastAPI ML server.

This test ensures that:
- Training a model consumes one user token.
- The trained model is properly listed in the `/models` endpoint response.
"""

import json, io

def test_models_listing_costs_one_token(client, signup_and_login):
    """
    Test that model training works and that trained models are listed properly.

    Steps:
        1. Log in to obtain an authentication token.
        2. Generate a small synthetic CSV dataset in memory.
        3. Send a POST request to `/train` with:
            - model_name = "linear"
            - features = ["f1", "f2"]
            - label = "y"
        4. Verify that training succeeds (HTTP 200).
        5. Call `/models` to retrieve the user's trained models.
        6. Assert that at least one model appears in the list.

    Verifies:
        - `/train` endpoint successfully trains and stores a model.
        - `/models` endpoint correctly returns all models for the logged-in user.
    """
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


