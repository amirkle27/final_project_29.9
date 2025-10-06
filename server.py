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
    return pwd_context.hash(p)

def verify_password(plain: str, hashed: str) -> bool:
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
    username: str
    password: str

class Login(BaseModel):
    username: str
    password: str

def _valid_password(p: str) -> bool:
    return (
        isinstance(p, str) and
        len(p) >= 6 and
        re.search(r"[A-Z]", p) and
        re.search(r"[^A-Za-z0-9]", p)
    )

def create_access_token(data: dict, minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    # שימוש ב־UTC עם timezone-aware כדי למנוע בעיות iat/exp
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=minutes)
    to_encode = data.copy()
    to_encode.update({"exp": expire, "iat": now})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def _save_upload_to_temp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename).suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read()); tmp.flush()
    return Path(tmp.name)

def _load_df(upload: UploadFile) -> pd.DataFrame:
    p = _save_upload_to_temp(upload)
    converter = FileConverterFactory().get(p)
    csv_path = converter.convert_to_csv(p)
    return pd.read_csv(csv_path)

def _parse_json_field(s: str) -> dict:
    try:
        return json.loads(s) if s else {}
    except Exception:
        log_and_raise(InputInvalidJSON(details={"value": s[:200]}))

def _clean_float(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None

@app.post("/token")
def issue_token(form: OAuth2PasswordRequestForm = Depends()):
    user = dal.get_user(form.username)
    if not user or not verify_password(form.password, user["password_hash"]):
        log_and_raise(AuthBadCredentials())
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/signup")
def signup(payload: Signup):
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
    user = dal.get_user(payload.username)
    if not user or not verify_password(payload.password, user["password_hash"]):
        log_and_raise(AuthBadCredentials())
    token = create_access_token({"sub": user["username"]})
    logger.info(f"User '{payload.username}' logged in")
    return {"access_token": token, "token_type": "bearer"}




def get_current_user(token: str = Depends(oauth2_scheme)):
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

        # חיוב 5 טוקנים + לוג שימוש
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
    missing = [c for c in features + [label] if c not in df.columns]
    if missing:
        log_and_raise(InputMissingColumns(details={"missing": missing}))

def _subset_df(df: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    # שומרים רק את הפיצ'רים והלייבל (כדי שה-Preprocessor שלך יעבוד בדיוק על מה שביקשו)
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
    """
    מאמן מודל, שומר אותו כ-pkl, רושם מטא-דאטה ב-DB, מחייב 1 טוקן, ומחזיר model_id.
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
    """
    מחזיר את כל המודלים ששמורים למשתמש המחובר.
    """
    models = dal.list_models(user["username"])
    # פעולה זו היא “מטא־דאטה” – תרצה לחייב 1 טוקן? (לפי הדרישה – אפשר)
    try:
        dal.update_tokens_and_log(
            username=user["username"],
            cost=1,  # אם אתה רוצה לחייב גם על קריאת מטא־דאטה
            action="list_models",
            model_name=None,
            file_name=None,
        )
        logger.info(f"User '{user['username']}' listed models")

    except ValueError:
        # לא נכשיל את הבקשה על רשימת מודלים — לשיקולך. אפשר גם כן לזרוק 402.
        pass

    return {"models": models}

@app.post("/predict/by_id/{model_id}")
def predict_by_id(
    model_id: str,
    data: str = Form(...),  # JSON dict: {"age":35, "salary":70000, "rooms":3}
    user: dict = Depends(get_current_user),
):
    try:
        # 1) שליפת מטא־דאטה ובעלות
        meta = dal.get_model(model_id)
        if not meta or meta["username"] != user["username"]:
            log_and_raise(ModelNotFound(details={"model_id": model_id}))

        # 2) טעינת ה-bundle
        bundle = joblib.load(meta["path"])
        model = bundle.get("model")
        if model is None:
            log_and_raise(ModelBundleCorrupt(details={"path": meta["path"]}))
        feature_cols = bundle.get("feature_cols")
        if not feature_cols:
            log_and_raise(ModelBundleCorrupt(details={"missing": "feature_cols"}))

        # 3) פרסינג של הנתונים החדשים + יישור לעמודות
        try:
            new_row = json.loads(data)
            if not isinstance(new_row, dict):
                raise ValueError("data is not a JSON object")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"data must be a JSON object of features: {e}")

        X_new = pd.DataFrame([new_row]).reindex(columns=feature_cols, fill_value=0)

        # 4) טרנספורמציות אם קיימות
        scaler = bundle.get("scaler")
        if scaler is not None:
            X_new = pd.DataFrame(scaler.transform(X_new), columns=feature_cols)

        poly = bundle.get("poly")
        if poly is not None:
            # poly מחזיר ndarray; המודל כבר מאומן עליו
            X_new = poly.transform(X_new)

        # 5) ניבוי
        y_hat = model.predict(X_new)

        if meta["kind"] == "classification":
            pred_value = y_hat[0] if hasattr(y_hat, "__len__") else y_hat
            resp = {"prediction": str(pred_value)}
        else:
            val = float(y_hat[0] if hasattr(y_hat, "__len__") else y_hat)
            resp = {"prediction": val}

        # 6) חיוב 5 טוקנים + לוג
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
            # DAL משליך ValueError עם "not enough tokens" כשאין מספיק
            raise HTTPException(status_code=402, detail=str(e))

        return resp

    except HTTPException:
        raise
    except Exception as e:
        # שמרנו ב-400 כדי לא לחשוף פרטים מיותרים, אבל כן להחזיר למזמין מה לא עבד
        raise HTTPException(status_code=400, detail=str(e))

class RemoveUser(BaseModel):
    username: str
    password: str

@app.delete("/remove_user")
def remove_user(payload: RemoveUser, user: dict = Depends(get_current_user)):
    # מאפשר למחוק רק את עצמך (או הרחב כאן לכללים אחרים)
    if payload.username != user["username"]:
        raise HTTPException(status_code=403, detail="Not allowed to remove other users")

    u = dal.get_user(payload.username)
    if not u or not verify_password(payload.password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="Bad credentials")

    # מחיקה ישירה ב-SQLite (מינימלי, בלי הרחבת DAL)
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
    if username != user["username"]:
        raise HTTPException(status_code=403, detail="Not allowed to view other users' tokens")

    u = dal.get_user(username)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")

    return {"tokens": int(u["tokens"])}


class AddTokens(BaseModel):
    username: str
    credit_card: str
    amount: int

@app.post("/add_tokens")
def add_tokens(payload: AddTokens, user: dict = Depends(get_current_user)):
    if payload.username != user["username"]:
        raise HTTPException(status_code=403, detail="Not allowed to add tokens to other users")

    if payload.amount <= 0:
        raise HTTPException(status_code=400, detail="amount must be positive")

    u = dal.get_user(payload.username)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")

    new_total = int(u["tokens"]) + int(payload.amount)
    dal.update_tokens(payload.username, new_total)
    # רישום בלוג שימוש (לא מחייב טוקן)
    dal.log_usage(payload.username, "add_tokens", None, None, tokens_after_usage=new_total)
    logger.info(f"User '{payload.username}' added {payload.amount} tokens (new total: {new_total})")

    return {"username": payload.username, "tokens": new_total}


@app.post("/predict/{model_name}")
def predict_model_name(
    model_name: str,
    data: dict = Body(...),
    user: dict = Depends(get_current_user),
):
    # 1) לאסוף את המודל האחרון של המשתמש עבור model_name
    models = dal.list_models(user["username"])  # כבר ממויין יורד לפי created_at
    meta = next((m for m in models if m["model_name"] == model_name), None)
    if not meta:
        raise HTTPException(status_code=404, detail=f"No trained model found for '{model_name}'")

    # 2) לטעון bundle וליישר פיצ'רים
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

    # 3) להכין נתונים חדשים
    try:
        X_new = pd.DataFrame([data]).reindex(columns=feature_cols, fill_value=0)
        if scaler is not None:
            X_new = pd.DataFrame(scaler.transform(X_new), columns=feature_cols)
        if poly is not None:
            X_new = poly.transform(X_new)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad input data: {e}")

    # 4) ניבוי + חיוב טוקנים
    try:
        y_hat = model.predict(X_new)
        if meta["kind"] == "classification":
            pred_value = y_hat[0] if hasattr(y_hat, "__len__") else y_hat
            response = {"prediction": str(pred_value)}
        else:
            val = float(y_hat[0] if hasattr(y_hat, "__len__") else y_hat)
            response = {"prediction": val}

        # חיוב 5 טוקנים
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
    reload = os.getenv("RELOAD", "0") == "1"  # פתחו רילוד בפיתוח: RELOAD=1
    workers = int(os.getenv("WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info")

    # אפשר להשתמש במחרוזת "module:app" או באובייקט app עצמו – שניהם תקינים.
    uvicorn.run("server:app", host=host, port=port, reload=reload, workers=workers, log_level=log_level)
    # לחלופין:
    # uvicorn.run(app, host=host, port=port, reload=reload, workers=workers, log_level=log_level)