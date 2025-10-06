"""
Data Access Layer (DAL) for the FastAPI ML server.

This module manages:
- SQLite connection and schema creation.
- CRUD operations for users and models.
- Token accounting (deduct/update) and usage logging.

Tables:
    users(username PK, password_hash, tokens, joined_at, usage_count)
    usage_logs(id PK, username, action, model_name, file_name, tokens_after_usage, created_at)
    models(id PK, username, model_name, kind, path, features JSON, label, metrics JSON, created_at)
"""

import sqlite3
from datetime import datetime
import json

DB_NAME = "ml_server.db"

def _connect():
    """
    Open a new SQLite connection to the configured database.

    Returns:
        sqlite3.Connection: An open connection to the database.
    """
    return sqlite3.connect(DB_NAME)

def init_db():
    """
    Initialize database schema (idempotent).
    Creates `users`, `usage_logs`, and `models` tables if they don't exist.
    """
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
                     """
    Insert a trained model record.

    Args:
        model_id: Unique identifier for the model.
        username: Owner username.
        model_name: Algorithm name (e.g., 'linear', 'logreg').
        kind: Problem type ('regression' / 'classification' / etc.).
        path: Filesystem path to the serialized model.
        features: List of feature column names used for training.
        label: Target column name.
        metrics: Training/evaluation metrics as a dict (stored as JSON).
    """
    with _connect() as conn:
        db = conn.cursor()
        db.execute("""
            INSERT INTO models(id, username, model_name, kind, path, features, label, metrics, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (model_id, username, model_name, kind, path,
              json.dumps(features), label, json.dumps(metrics), datetime.now()))
        conn.commit()

def list_models(username: str):
    """
    List models that belong to a given user.

    Args:
        username: The owner's username.

    Returns:
        list[dict]: Model rows with JSON fields parsed into Python objects.
    """
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
    """
    Fetch a single model record by ID.

    Args:
        model_id: The model's unique identifier.

    Returns:
        dict | None: Parsed model record, or None if not found.
    """
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
    """
    Create a new user.

    Args:
        username: Unique username (PK).
        password_hash: Hashed password.
        initial_tokens: Starting token balance (default 15).
    """
    with _connect() as conn:
        db = conn.cursor()
        db.execute(
            "INSERT INTO users(username, password_hash, tokens, joined_at, usage_count) VALUES(?,?,?,?,?)",
            (username, password_hash, initial_tokens, datetime.now(),0))
        conn.commit()

def get_user(username: str):
    """
    Retrieve a user by username.

    Args:
        username: Username to look up.

    Returns:
        dict | None: User record with fields (username, password_hash, tokens, joined_at, usage_count),
                     or None if the user doesn't exist.
    """
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
    """
    Set a user's token balance to a specific value.

    Args:
        username: The user to update.
        new_tokens: New token count (must be >= 0).
    """
    with _connect() as conn:
        db = conn.cursor()
        db.execute("UPDATE users SET tokens=? WHERE username=?", (new_tokens, username))
        conn.commit()

def increment_usage(username: str):
    """
    Increment the usage counter for a user by 1.

    Args:
        username: The user to update.
    """
    with _connect() as conn:
        db = conn.cursor()
        db.execute("UPDATE users SET usage_count = usage_count + 1 WHERE username=?", (username,))
        conn.commit()

def log_usage(username: str, action: str, model_name: str | None, file_name: str | None, tokens_after_usage: int):
    """
    Insert a usage log entry.

    Args:
        username: The acting user.
        action: Action string (e.g., 'train', 'predict/by_id').
        model_name: Optional model name involved in the action.
        file_name: Optional uploaded file name involved in the action.
        tokens_after_usage: User's token balance after the action.
    """
    with _connect() as conn:
        db = conn.cursor()
        db.execute("""
            INSERT INTO usage_logs(username, action, model_name, file_name, tokens_after_usage, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, action, model_name, file_name, tokens_after_usage, datetime.now()))
        conn.commit()

def update_tokens_and_log(username: str, cost: int, action: str, model_name: str | None, file_name: str | None):
    """
    Atomically deduct tokens for an action and write a usage log entry.

    Args:
        username: Acting user.
        cost: Positive integer cost in tokens.
        action: Action label (e.g., 'train', 'predict/by_id').
        model_name: Optional model name.
        file_name: Optional file name.
    Returns:
        int: New token balance after deduction.

    Raises:
        ValueError: If user is unknown, cost is invalid, or user lacks tokens.
    """
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


