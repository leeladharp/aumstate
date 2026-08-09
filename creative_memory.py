from __future__ import annotations

import sqlite3
from datetime import datetime


DB_PATH = "aumstate_memory.db"


def init_creative_memory_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS creative_preferences (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS creative_projects (
            project_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content_type TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS creative_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            relatability INTEGER,
            humor INTEGER,
            depth INTEGER,
            preachiness INTEGER,
            notes TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def save_creative_preference(key: str, value: str, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO creative_preferences (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (key, value, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def load_creative_preferences(db_path: str = DB_PATH) -> dict[str, str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT key, value
        FROM creative_preferences
        ORDER BY key ASC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return {key: value for key, value in rows}


def create_creative_project(project_id: str, title: str, content_type: str, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO creative_projects (project_id, title, content_type, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (project_id, title, content_type, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def save_creative_feedback(
    project_id: str,
    relatability: int | None,
    humor: int | None,
    depth: int | None,
    preachiness: int | None,
    notes: str,
    db_path: str = DB_PATH,
) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO creative_feedback (
            project_id,
            relatability,
            humor,
            depth,
            preachiness,
            notes,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            relatability,
            humor,
            depth,
            preachiness,
            notes,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def load_recent_creative_feedback(limit: int = 10, db_path: str = DB_PATH) -> list[dict[str, str | int | None]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT project_id, relatability, humor, depth, preachiness, notes, created_at
        FROM creative_feedback
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "project_id": row[0],
            "relatability": row[1],
            "humor": row[2],
            "depth": row[3],
            "preachiness": row[4],
            "notes": row[5],
            "created_at": row[6],
        }
        for row in rows
    ]


def build_role_memory(role: str, preferences: dict[str, str], feedback_items: list[dict[str, str | int | None]]) -> str:
    role_keys = {
        "psychology": ["preferred_topics"],
        "philosophy": ["preferred_ending_style", "avoid_preachy"],
        "ambiguity": ["preferred_topics", "avoid_preachy"],
        "humor": ["preferred_humor_style"],
        "story": ["preferred_visual_style", "preferred_ending_style"],
        "critic": ["avoid_preachy", "preferred_ending_style"],
    }

    lines: list[str] = []
    for key in role_keys.get(role, []):
        value = preferences.get(key)
        if value:
            lines.append(f"- {key}: {value}")

    if role == "humor":
        for item in feedback_items[:3]:
            if item.get("humor") is not None or item.get("notes"):
                lines.append(
                    f"- feedback: humor={item.get('humor')} notes={item.get('notes')}"
                )

    if role in {"philosophy", "critic"}:
        for item in feedback_items[:3]:
            if item.get("depth") is not None or item.get("preachiness") is not None:
                lines.append(
                    f"- feedback: depth={item.get('depth')} preachiness={item.get('preachiness')} notes={item.get('notes')}"
                )

    return "\n".join(lines).strip()
