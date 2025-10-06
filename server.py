"""
FastAPI ML Server

Provides authentication (signup/login/JWT), token accounting, model training,
prediction, and model management. Models are trained via facades from
`model_factory`, saved as joblib bundles under ./models, and metadata is stored
through the `dal` layer (SQLite).

Main endpoints:
- POST /signup, POST /login, POST /token
- GET  /health
- POST /train
- GET  /models
- POST /predict/classification
- POST /predict/regression
- POST /predict/by_id/{model_id}
- POST /predict/{model_name}     (use latest trained model of that name)
- GET  /tokens/{username}
- POST /add_tokens
- DELETE /remove_user
"""

import dal
import jwt
from enum import Enum
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Body
from pathlib import Path
import pandas as pd
import tempfile, json
import sqlite3
import re
from pydantic import BaseModel
import os, uuid, joblib
import logging
from logging.handlers import RotatingFileHandler
from File_Converter_Factory import FileConverterFactory
from model_factory import ModelFactory
from datetime import datetime, timedelta, timezone
from fastapi import Depends, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
import math
from errors import (
    AuthBadCredentials, AuthUserNotFound, AuthTokenExpired, AuthTokenInvalid,
    InputInvalidJSON, InputMissingColumns, InputModelParamsInvalid,
    TokensNotEnough, ModelNotFound, ModelKindMismatch, ModelTrainFailed,
    ModelPredictFailed, ModelBundleCorrupt, InternalError, log_and_raise
)
from http import HTTPStatus

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p: str) -> str:
    """Return a bcrypt hash for a plaintext password."""
    return pwd_context.hash(p)

def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    return pwd_context.verify(plain, hashed)

#SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
SECRET_KEY = "MY_SUPER_FIXED_SECRET_123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


dal.init_db()
app = FastAPI(title="ML Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---- Logging setup (rotate at ~5MB, keep 3 backups) ----
logger = logging.getLogger("ml_server")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = RotatingFileHandler("server.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)

logger.info("Server starting up")


class Signup(BaseModel):
    """Request body for /signup."""
    username: str
    password: str

class Login(BaseModel):
    """Request body for /login (username/password)."""
    username: str
    password: str

def _valid_password(p: str) -> bool:
    """Minimal password policy: ≥6 chars, one uppercase, one special char."""
    return (
        isinstance(p, str) and
        len(p) >= 6 and
        re.search(r"[A-Z]", p) and
        re.search(r"[^A-Za-z0-9]", p)
    )

def _create_access_token(data: dict, minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    """Create a signed JWT with `sub` claim and exp/iat in UTC."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=minutes)
    to_encode = data.copy()
    to_encode.update({"exp": expire, "iat": now})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def _save_upload_to_temp(upload: UploadFile) -> Path:
    """Persist an uploaded file to a temp path and return its Path."""
    suffix = Path(upload.filename).suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read()); tmp.flush()
    return Path(tmp.name)

def _load_df(upload: UploadFile) -> pd.DataFrame:
    """Convert uploaded file (csv/xlsx/json…) to CSV via factory and read as DataFrame."""
    p = _save_upload_to_temp(upload)
    converter = FileConverterFactory().get(p)
    csv_path = converter.convert_to_csv(p)
    return pd.read_csv(csv_path)

def _parse_json_field(s: str) -> dict:
    """Parse a JSON string field to dict; raise InputInvalidJSON on failure."""
    try:
        return json.loads(s) if s else {}
    except Exception:
        log_and_raise(InputInvalidJSON(details={"value": s[:200]}))

def _clean_float(x):
    """Safe float conversion; returns None if NaN/Inf or conversion fails."""
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None

@app.get("/health")
def health():
    """Liveness probe for the service."""
    return {"ok": True}
    
@app.post("/token")
def issue_token(form: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 token endpoint. Validates credentials and returns a bearer JWT."""
    user = dal.get_user(form.username)
    if not user or not verify_password(form.password, user["password_hash"]):
        log_and_raise(AuthBadCredentials())
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/signup")
def signup(payload: Signup):
    """Create a new user and seed with initial tokens."""
    if not _valid_password(payload.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password too weak: min 6 chars, one uppercase, one special char."
        )
    try:
        dal.insert_user(payload.username, hash_password(payload.password), initial_tokens=15)
        dal.log_usage(payload.username, "signup", None, None, tokens_after_usage=15)
        logger.info(f"User '{payload.username}' registered")
        return {"message": "user created", "username": payload.username, "tokens": 15}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="username already exists")

@app.post("/login")
def login(payload: Login):
    """Login with username/password and receive a bearer JWT."""
    user = dal.get_user(payload.username)
    if not user or not verify_password(payload.password, user["password_hash"]):
        log_and_raise(AuthBadCredentials())
    token = create_access_token({"sub": user["username"]})
    logger.info(f"User '{payload.username}' logged in")
    return {"access_token": token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency that validates the JWT and returns the user record."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            log_and_raise(AuthTokenInvalid())
    except jwt.ExpiredSignatureError:
        log_and_raise(AuthTokenExpired())
    except jwt.InvalidTokenError:
        log_and_raise(AuthTokenInvalid())

    user = dal.get_user(username)
    if not user:
        log_and_raise(AuthUserNotFound())
    return user

@app.post("/predict/classification")
def predict_classification(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    model: str = Form("logreg"),
    data: str = Form("{}"),
    model_params: str = Form("{}"),
    user: dict = Depends(get_current_user),
):
    """Train a classifier on the uploaded dataset and return a single prediction.

    Body (multipart/form-data):
      - file: dataset file (csv/xlsx/json…)
      - target_col: name of label column
      - model: model key for `ModelFactory` (default: 'logreg')
      - data: JSON of a single row to predict
      - model_params: JSON dict of hyper-params to pass into the facade

    Returns:
      model name, metrics (accuracy), and the predicted class as string.

    Charges 5 tokens on success.
    """
    try:
        df = _load_df(file)
        new_row = _parse_json_field(data)
        params = _parse_json_field(model_params)

        facade, kind = ModelFactory.create(model, params)
        if kind != "classification":
            log_and_raise(ModelKindMismatch(details={"expected": "classification", "got": kind}))

        results = facade.train_and_evaluate(df, target_col=target_col)
        pred = facade.predict(pd.DataFrame([new_row]))

        if hasattr(pred, "iloc"):
            pred_value = pred.iloc[0]
        elif isinstance(pred, (list, tuple)):
            pred_value = pred[0]
        else:
            pred_value = pred
            
        try:
            dal.update_tokens_and_log(
                username=user["username"],
                cost=5,
                action="predict/classification",
                model_name=model,
                file_name=file.filename,
            )
        except ValueError as e:
            if "not enough tokens" in str(e).lower():
                log_and_raise(TokensNotEnough())
            else:
                log_and_raise(InternalError(details={"cause": str(e)}))

        return {
            "model": model,
            "target_col": target_col,
            "prediction": str(pred_value),
            "metrics": {"accuracy": _clean_float(results.get("accuracy"))},
            "used_params": params,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/predict/regression")
def predict_regression(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    model: str = Form("linear"),
    data: str = Form("{}"),
    model_params: str = Form("{}"),
    user: dict = Depends(get_current_user),
):
    """Train a regressor on the uploaded dataset and return a single prediction.

    Same contract as /predict/classification, but returns numeric metrics
    (mse, rmse, r2) and a float prediction. Charges 5 tokens on success.
    """
    try:
        df = _load_df(file)
        new_row = _parse_json_field(data)
        params = _parse_json_field(model_params)

        facade, kind = ModelFactory.create(model, params)
        if kind != "regression":
            log_and_raise(ModelKindMismatch(details={"expected": "regression", "got": kind}))

        results = facade.train_and_evaluate(df, target_col=target_col)
        pred = facade.predict(pd.DataFrame([new_row]))

        pred_value = float(pred.iloc[0]) if hasattr(pred, "iloc") else (
            float(pred[0]) if isinstance(pred, (list, tuple)) else float(pred)
        )

        try:
            dal.update_tokens_and_log(
                username=user["username"],
                cost=5,
                action="predict/regression",
                model_name=model,
                file_name=file.filename,
            )
        except ValueError as e:
            if "not enough tokens" in str(e).lower():
                log_and_raise(TokensNotEnough())
            else:
                log_and_raise(InternalError(details={"cause": str(e)}))

        return {
            "model": model,
            "target_col": target_col,
            "prediction": pred_value,
            "metrics": {
                "mse": _clean_float(results.get("mse")),
                "rmse": _clean_float(results.get("rmse")),
                "r2": _clean_float(results.get("r2")),
            },
            "used_params": params,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))

def _ensure_features(df: pd.DataFrame, features: list[str], label: str):
    """Validate that all requested features + label exist in the dataframe."""
    missing = [c for c in features + [label] if c not in df.columns]
    if missing:
        log_and_raise(InputMissingColumns(details={"missing": missing}))

def _subset_df(df: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    """Return a copy containing only selected features + label."""
    return df[features + [label]].copy()

@app.post("/train")
def train_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    features: str = Form(...),      # JSON list: ["age","salary","rooms"] או "age,salary,rooms"
    label: str = Form(...),
    model_params: str = Form("{}"), # JSON dict
    user: dict = Depends(get_current_user),
):
    """Train a model and persist its bundle to disk + metadata to DB.

    Request:
      - file: dataset file
      - model_name: key supported by ModelFactory (e.g. 'linear', 'logreg'…)
      - features: JSON array or comma-separated list
      - label: target column name
      - model_params: JSON dict of hyper-parameters

    Returns:
      model_id, metrics and saved metadata. Charges 1 token on success.
    """
    try:
        # --- parse features (JSON array or comma-separated) ---
        try:
            features_list = json.loads(features)
            if not isinstance(features_list, list):
                raise ValueError("features not a JSON array")
        except Exception:
            features_list = [c.strip() for c in re.split(r'[,\s]+', features) if c.strip()]
            if not features_list:
                raise HTTPException(
                    status_code=400,
                    detail='features must be JSON array (e.g. ["age","weight"]) or comma-separated (age,weight)'
                )

        # --- parse model_params ---
        try:
            params = json.loads(model_params) if model_params else {}
            if not isinstance(params, dict):
                raise ValueError("model_params not a JSON object")
        except Exception:
            raise HTTPException(status_code=400, detail="model_params must be a JSON object, e.g. {}")

        # --- load & validate data ---
        df = _load_df(file)
        _ensure_features(df, features_list, label)
        df = _subset_df(df, features_list, label)

        # >>> חיוב טוקן בראש — כאן <<<
        try:
            dal.update_tokens_and_log(
                username=user["username"],
                cost=1,                # אימון = 1 טוקן
                action="train",
                model_name=model_name,
                file_name=file.filename,
            )
        except ValueError as e:
            logger.warning(f"User '{user['username']}' tried to train without enough tokens")
            raise HTTPException(status_code=402, detail=str(e))

        # --- build facade ---
        facade, kind = ModelFactory.create(model_name, params)

        # --- train & evaluate ---
        results = facade.train_and_evaluate(df, target_col=label)

        # --- bundle to save ---
        bundle = {
            "model_name": model_name,
            "kind": kind,
            "facade_class": facade.__class__.__name__,
            "model": getattr(facade, "model", None),
            "scaler": getattr(facade, "scaler", None),
            "poly": getattr(facade, "poly", None),
            "feature_cols": list(getattr(facade, "feature_cols", features_list)),
            "label": label,
        }

        # --- persist ---
        model_id = str(uuid.uuid4())
        os.makedirs("models", exist_ok=True)
        pkl_path = os.path.join("models", f"{model_id}.pkl")
        joblib.dump(bundle, pkl_path)

        # --- metrics (ניקוי NaN/Inf) ---
        if kind == "classification":
            metrics = {"accuracy": _clean_float(results.get("accuracy"))}
        else:
            metrics = {
                "mse":  _clean_float(results.get("mse")),
                "rmse": _clean_float(results.get("rmse")),
                "r2":   _clean_float(results.get("r2")),
            }

        # --- save metadata in DB ---
        dal.insert_model(
            model_id=model_id,
            username=user["username"],
            model_name=model_name,
            kind=kind,
            path=pkl_path,
            features=features_list,
            label=label,
            metrics=metrics,
        )

        logger.info(f"User '{user['username']}' trained model='{model_name}' label='{label}' features={features_list}")

        return {
            "status": "model trained",
            "model_id": model_id,
            "model_name": model_name,
            "kind": kind,
            "features": features_list,
            "label": label,
            "metrics": metrics,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/models")
def list_user_models(user: dict = Depends(get_current_user)):
    """Return all models owned by the current user (optionally charges 1 token)."""
    models = dal.list_models(user["username"])
    try:
        dal.update_tokens_and_log(
            username=user["username"],
            cost=1,  
            action="list_models",
            model_name=None,
            file_name=None,
        )
        logger.info(f"User '{user['username']}' listed models")

    except ValueError:
        pass

    return {"models": models}

@app.post("/predict/by_id/{model_id}")
def predict_by_id(
    model_id: str,
    data: str = Form(...),  # JSON dict: {"age":35, "salary":70000, "rooms":3}
    user: dict = Depends(get_current_user),
):
    """Predict using a previously trained/saved model bundle by its model_id.

    Body:
      - data: JSON object whose keys match the trained feature columns.

    Returns:
      {"prediction": <str or float>} and charges 5 tokens.
    """
    try:
        meta = dal.get_model(model_id)
        if not meta or meta["username"] != user["username"]:
            log_and_raise(ModelNotFound(details={"model_id": model_id}))
            
        bundle = joblib.load(meta["path"])
        model = bundle.get("model")
        if model is None:
            log_and_raise(ModelBundleCorrupt(details={"path": meta["path"]}))
        feature_cols = bundle.get("feature_cols")
        if not feature_cols:
            log_and_raise(ModelBundleCorrupt(details={"missing": "feature_cols"}))

        try:
            new_row = json.loads(data)
            if not isinstance(new_row, dict):
                raise ValueError("data is not a JSON object")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"data must be a JSON object of features: {e}")

        X_new = pd.DataFrame([new_row]).reindex(columns=feature_cols, fill_value=0)

        scaler = bundle.get("scaler")
        if scaler is not None:
            X_new = pd.DataFrame(scaler.transform(X_new), columns=feature_cols)

        poly = bundle.get("poly")
        if poly is not None:
            X_new = poly.transform(X_new)

        y_hat = model.predict(X_new)

        if meta["kind"] == "classification":
            pred_value = y_hat[0] if hasattr(y_hat, "__len__") else y_hat
            resp = {"prediction": str(pred_value)}
        else:
            val = float(y_hat[0] if hasattr(y_hat, "__len__") else y_hat)
            resp = {"prediction": val}

        try:
            dal.update_tokens_and_log(
                username=user["username"],
                cost=5,
                action="predict/by_id",
                model_name=meta["model_name"],
                file_name=None,
            )
            logger.info(f"User '{user['username']}' predicted by_id model_id='{model_id}'")
        except ValueError as e:
            raise HTTPException(status_code=402, detail=str(e))

        return resp

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class RemoveUser(BaseModel):
    """Body for DELETE /remove_user."""
    username: str
    password: str

@app.delete("/remove_user")
def remove_user(payload: RemoveUser, user: dict = Depends(get_current_user)):
    """Delete the current user's account and all related data (self-service only)."""
    if payload.username != user["username"]:
        raise HTTPException(status_code=403, detail="Not allowed to remove other users")

    u = dal.get_user(payload.username)
    if not u or not verify_password(payload.password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="Bad credentials")

    with sqlite3.connect(dal.DB_NAME) as conn:
        db = conn.cursor()
        db.execute("DELETE FROM models WHERE username=?", (payload.username,))
        db.execute("DELETE FROM usage_logs WHERE username=?", (payload.username,))
        db.execute("DELETE FROM users WHERE username=?", (payload.username,))
        conn.commit()
        logger.warning(f"User '{payload.username}' removed account")

    return {"message": f"user '{payload.username}' removed"}

@app.get("/tokens/{username}")
def get_tokens(username: str, user: dict = Depends(get_current_user)):
    """Return current token balance for the authenticated user."""
    if username != user["username"]:
        raise HTTPException(status_code=403, detail="Not allowed to view other users' tokens")

    u = dal.get_user(username)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")

    return {"tokens": int(u["tokens"])}


class AddTokens(BaseModel):
    """Body for POST /add_tokens (mock billing)."""
    username: str
    credit_card: str
    amount: int

@app.post("/add_tokens")
def add_tokens(payload: AddTokens, user: dict = Depends(get_current_user)):
    """Increase the authenticated user's token balance (demo payment endpoint)."""
    if payload.username != user["username"]:
        raise HTTPException(status_code=403, detail="Not allowed to add tokens to other users")

    if payload.amount <= 0:
        raise HTTPException(status_code=400, detail="amount must be positive")

    u = dal.get_user(payload.username)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")

    new_total = int(u["tokens"]) + int(payload.amount)
    dal.update_tokens(payload.username, new_total)
    dal.log_usage(payload.username, "add_tokens", None, None, tokens_after_usage=new_total)
    logger.info(f"User '{payload.username}' added {payload.amount} tokens (new total: {new_total})")

    return {"username": payload.username, "tokens": new_total}


@app.post("/predict/{model_name}")
def predict_model_name(
    model_name: str,
    data: dict = Body(...),
    user: dict = Depends(get_current_user),
):
    """Predict using the most recent model the user trained with this `model_name`.

    Body:
      - data: JSON object whose keys match the trained feature columns.
    """
    models = dal.list_models(user["username"])  # כבר ממויין יורד לפי created_at
    meta = next((m for m in models if m["model_name"] == model_name), None)
    if not meta:
        raise HTTPException(status_code=404, detail=f"No trained model found for '{model_name}'")

    try:
        bundle = joblib.load(meta["path"])
        feature_cols = bundle["feature_cols"]
        scaler = bundle.get("scaler")
        poly = bundle.get("poly")
        model = bundle.get("model")
        if model is None:
            raise HTTPException(status_code=500, detail="Saved model bundle is missing 'model'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model bundle: {e}")

    try:
        X_new = pd.DataFrame([data]).reindex(columns=feature_cols, fill_value=0)
        if scaler is not None:
            X_new = pd.DataFrame(scaler.transform(X_new), columns=feature_cols)
        if poly is not None:
            X_new = poly.transform(X_new)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad input data: {e}")

    try:
        y_hat = model.predict(X_new)
        if meta["kind"] == "classification":
            pred_value = y_hat[0] if hasattr(y_hat, "__len__") else y_hat
            response = {"prediction": str(pred_value)}
        else:
            val = float(y_hat[0] if hasattr(y_hat, "__len__") else y_hat)
            response = {"prediction": val}

        try:
            dal.update_tokens_and_log(
                username=user["username"],
                cost=5,
                action=f"predict/{model_name}",
                model_name=model_name,
                file_name=None,
            )
            logger.info(f"User '{user['username']}' predicted with model='{model_name}' kind='{meta['kind']}'")

        except ValueError as e:
            logger.warning(f"User '{user['username']}' tried to predict without enough tokens")
            raise HTTPException(status_code=402, detail=str(e))

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "0") == "1"  
    workers = int(os.getenv("WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info")

    uvicorn.run("server:app", host=host, port=port, reload=reload, workers=workers, log_level=log_level)
