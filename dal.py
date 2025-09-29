# dal.py
import sqlite3
from datetime import datetime
import json

DB_NAME = "ml_server.db"

def _connect():
    return sqlite3.connect(DB_NAME)

def init_db():
    with _connect() as conn:
        db = conn.cursor()

        db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    tokens INTEGER NOT NULL,
                    joined_at TEXT NOT NULL,
                    usage_count INTEGER NOT NULL
                )
                """)
        conn.commit()
        db.execute("""
                CREATE TABLE IF NOT EXISTS usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    model_name TEXT,
                    file_name TEXT,
                    tokens_after_usage INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """)
        conn.commit()
        db.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            model_name TEXT NOT NULL,   
            kind TEXT NOT NULL,         
            path TEXT NOT NULL,        
            features TEXT NOT NULL,    
            label TEXT NOT NULL,
            metrics TEXT NOT NULL,      
            created_at TEXT NOT NULL
        )
        """)
        conn.commit()

def insert_model(model_id: str, username: str, model_name: str, kind: str,
                 path: str, features: list[str], label: str, metrics: dict):
    with _connect() as conn:
        db = conn.cursor()
        db.execute("""
            INSERT INTO models(id, username, model_name, kind, path, features, label, metrics, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (model_id, username, model_name, kind, path,
              json.dumps(features), label, json.dumps(metrics), datetime.now()))
        conn.commit()

def list_models(username: str):
    with _connect() as conn:
        db = conn.cursor()
        rows = db.execute("""
            SELECT id, model_name, kind, path, features, label, metrics, created_at
            FROM models WHERE username=? ORDER BY created_at DESC
        """, (username,)).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r[0],
                "model_name": r[1],
                "kind": r[2],
                "path": r[3],
                "features": json.loads(r[4]),
                "label": r[5],
                "metrics": json.loads(r[6]),
                "created_at": r[7],
            })
        return out

def get_model(model_id: str):
    with _connect() as conn:
        db = conn.cursor()
        r = db.execute("""
            SELECT id, username, model_name, kind, path, features, label, metrics, created_at
            FROM models WHERE id=?
        """, (model_id,)).fetchone()
        if not r:
            return None
        return {
            "id": r[0],
            "username": r[1],
            "model_name": r[2],
            "kind": r[3],
            "path": r[4],
            "features": json.loads(r[5]),
            "label": r[6],
            "metrics": json.loads(r[7]),
            "created_at": r[8],
        }

def insert_user(username: str, password_hash: str, initial_tokens: int = 15):
    with _connect() as conn:
        db = conn.cursor()
        db.execute(
            "INSERT INTO users(username, password_hash, tokens, joined_at, usage_count) VALUES(?,?,?,?,?)",
            (username, password_hash, initial_tokens, datetime.now(),0))
        conn.commit()

def get_user(username: str):
    with _connect() as conn:
        db = conn.cursor()
        row = db.execute("SELECT username, password_hash, tokens, joined_at, usage_count FROM users WHERE username=?", (username,)).fetchone()
        if not row:
            return None
        return {
            "username": row[0],
            "password_hash": row[1],
            "tokens": row[2],
            "joined_at": row[3],
            "usage_count": row[4],
        }

def update_tokens(username: str, new_tokens: int):
    with _connect() as conn:
        db = conn.cursor()
        db.execute("UPDATE users SET tokens=? WHERE username=?", (new_tokens, username))
        conn.commit()

def increment_usage(username: str):
    with _connect() as conn:
        db = conn.cursor()
        db.execute("UPDATE users SET usage_count = usage_count + 1 WHERE username=?", (username,))
        conn.commit()

def log_usage(username: str, action: str, model_name: str | None, file_name: str | None, tokens_after_usage: int):
    with _connect() as conn:
        db = conn.cursor()
        db.execute("""
            INSERT INTO usage_logs(username, action, model_name, file_name, tokens_after_usage, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, action, model_name, file_name, tokens_after_usage, datetime.now()))
        conn.commit()

def update_tokens_and_log(username: str, cost: int, action: str, model_name: str | None, file_name: str | None):
    if not isinstance(cost, int) or cost <= 0:
        raise ValueError("cost must be a positive integer")

    with _connect() as conn:
        db = conn.cursor()

        row = db.execute("SELECT tokens FROM users WHERE username=?", (username,)).fetchone()
        if not row:
            raise ValueError("unknown user")
        tokens = int(row[0])
        if tokens < cost:
            raise ValueError("not enough tokens")

        new_tokens = tokens - cost

        db.execute(
            "UPDATE users SET tokens=?, usage_count=usage_count+1 WHERE username=?",
            (new_tokens, username)
        )
        db.execute(
            """
            INSERT INTO usage_logs(username, action, model_name, file_name, tokens_after_usage, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            (username, action, model_name, file_name, new_tokens, datetime.now())
        )
        conn.commit()

    return new_tokens
