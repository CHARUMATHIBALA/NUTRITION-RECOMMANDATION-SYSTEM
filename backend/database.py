# backend/database.py
"""SQLite helper for the Healthcare dashboard.

Provides a simple connection, table creation, and CRUD helpers for:
- users (id, username, name, password_hash)
- analysis_history (id, user_id, timestamp, patient_name, data JSON)

The module lazily creates tables on first import.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

# Database file placed at project root
DB_PATH = Path(__file__).resolve().parents[1] / "healthcare.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    cur = conn.cursor()
    # Users table (optional, not used in this minimal implementation)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT,
            password_hash TEXT
        )
        """
    )
    # Analysis history
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT NOT NULL,
            patient_name TEXT,
            data TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()
    conn.close()

# Initialise tables on import
init_db()


def save_analysis(user_id: int | None, patient_name: str, data: dict) -> None:
    """Persist analysis JSON for a given user (or None for anonymous)."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analysis_history (user_id, timestamp, patient_name, data) VALUES (?, ?, ?, ?)",
        (user_id, datetime.utcnow().isoformat(), patient_name, json.dumps(data)),
    )
    conn.commit()
    conn.close()


def get_history(user_id: int | None = None, limit: int = 100):
    """Retrieve analysis history rows as list of dicts."""
    conn = _get_conn()
    cur = conn.cursor()
    if user_id is None:
        cur.execute("SELECT * FROM analysis_history ORDER BY timestamp DESC LIMIT ?", (limit,))
    else:
        cur.execute(
            "SELECT * FROM analysis_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        )
    rows = cur.fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "timestamp": r["timestamp"],
            "patient_name": r["patient_name"],
            "data": json.loads(r["data"]),
        })
    return result
