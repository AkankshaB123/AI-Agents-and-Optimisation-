"""SQLite-backed conversation persistence for the Discovery Agent.

The database is intentionally small and local: Streamlit session state remains the
hot state, while SQLite provides durable state for the lifetime of the runtime.
Set DISCOVERY_DB_PATH to place the DB on a mounted/external volume in production.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "discovery_agent.db"
MAX_TEXT = 12000
MAX_JSON = 60000


def _db_path() -> Path:
    configured = os.getenv("DISCOVERY_DB_PATH", "").strip()
    path = Path(configured).expanduser() if configured else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any, limit: int = MAX_JSON) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text[:limit]


@contextmanager
def connection():
    conn = sqlite3.connect(_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                conversation_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (conversation_id, key),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                query TEXT NOT NULL,
                providers_json TEXT,
                results_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS agent_state (
                conversation_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, id);
            CREATE INDEX IF NOT EXISTS idx_search_history_conversation
                ON search_history(conversation_id, id);
            """
        )


def create_conversation(title: str | None = None, conversation_id: str | None = None) -> str:
    conversation_id = conversation_id or uuid.uuid4().hex
    now = _now()
    with connection() as conn:
        conn.execute(
            "INSERT INTO conversations(id, created_at, updated_at, title) VALUES (?, ?, ?, ?)",
            (conversation_id, now, now, title),
        )
    return conversation_id


def ensure_conversation(conversation_id: str) -> str:
    with connection() as conn:
        row = conn.execute(
            "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row:
            return conversation_id
    return create_conversation(conversation_id=conversation_id)


def touch_conversation(conversation_id: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")
    ensure_conversation(conversation_id)
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO messages(conversation_id, role, content, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, str(content)[:MAX_TEXT], _json(metadata or {}, 12000), _now()),
        )
    touch_conversation(conversation_id)


def load_messages(conversation_id: str, limit: int = 50) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, metadata_json, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    rows = list(reversed(rows))
    result = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        result.append(
            {
                "role": row["role"],
                "content": row["content"],
                "metadata": metadata,
                "created_at": row["created_at"],
            }
        )
    return result


def save_preferences(conversation_id: str, preferences: dict[str, Any]) -> None:
    ensure_conversation(conversation_id)
    now = _now()
    with connection() as conn:
        for key, value in preferences.items():
            conn.execute(
                """
                INSERT INTO user_preferences(conversation_id, key, value_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id, key)
                DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
                """,
                (conversation_id, str(key)[:200], _json(value, 12000), now),
            )
    touch_conversation(conversation_id)


def load_preferences(conversation_id: str) -> dict[str, Any]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT key, value_json FROM user_preferences WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
    result = {}
    for row in rows:
        try:
            result[row["key"]] = json.loads(row["value_json"])
        except json.JSONDecodeError:
            result[row["key"]] = row["value_json"]
    return result


def save_search(
    conversation_id: str,
    query: str,
    providers: list[str],
    results: list[dict[str, Any]],
) -> None:
    ensure_conversation(conversation_id)
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO search_history(
                conversation_id, query, providers_json, results_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                str(query)[:MAX_TEXT],
                _json(providers, 8000),
                _json(results, MAX_JSON),
                _now(),
            ),
        )
    touch_conversation(conversation_id)


def load_search_history(conversation_id: str, limit: int = 10) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 100))
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT query, providers_json, results_json, created_at
            FROM search_history
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    result = []
    for row in rows:
        try:
            providers = json.loads(row["providers_json"] or "[]")
        except json.JSONDecodeError:
            providers = []
        try:
            results = json.loads(row["results_json"] or "[]")
        except json.JSONDecodeError:
            results = []
        result.append(
            {
                "query": row["query"],
                "providers": providers,
                "results": results,
                "created_at": row["created_at"],
            }
        )
    return result


def save_agent_state(conversation_id: str, state: dict[str, Any]) -> None:
    ensure_conversation(conversation_id)
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_state(conversation_id, state_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(conversation_id)
            DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at
            """,
            (conversation_id, _json(state), _now()),
        )
    touch_conversation(conversation_id)


def load_agent_state(conversation_id: str) -> dict[str, Any]:
    with connection() as conn:
        row = conn.execute(
            "SELECT state_json FROM agent_state WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    if not row:
        return {}
    try:
        return json.loads(row["state_json"])
    except json.JSONDecodeError:
        return {}


def build_memory_context(conversation_id: str, message_limit: int = 12) -> dict[str, Any]:
    """Return compact, model-safe memory for the next agent turn."""
    messages = load_messages(conversation_id, limit=message_limit)
    return {
        "preferences": load_preferences(conversation_id),
        "recent_messages": [
            {"role": item["role"], "content": item["content"]} for item in messages
        ],
        "recent_searches": [
            {
                "query": item["query"],
                "providers": item["providers"],
            }
            for item in load_search_history(conversation_id, limit=5)
        ],
        "agent_state": load_agent_state(conversation_id),
    }
