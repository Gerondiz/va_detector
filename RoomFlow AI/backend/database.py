import sqlite3
import os
import time
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data.db")
FACE_DIR = os.path.join(BASE_DIR, "logs", "faces")

SETTINGS_CACHE: dict[str, str] = {}
_SETTINGS_DB_INIT = False


def _ensure_settings_table():
    global _SETTINGS_DB_INIT
    if _SETTINGS_DB_INIT:
        return
    conn = get_conn()
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""")
        conn.commit()
        _SETTINGS_DB_INIT = True
    finally:
        conn.close()


def get_setting(key: str, default: str = "") -> str:
    _ensure_settings_table()
    if key in SETTINGS_CACHE:
        return SETTINGS_CACHE[key]
    conn = get_conn()
    try:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        val = row["value"] if row else default
        SETTINGS_CACHE[key] = val
        return val
    finally:
        conn.close()


def set_setting(key: str, value: str):
    _ensure_settings_table()
    SETTINGS_CACHE[key] = value
    conn = get_conn()
    try:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
    finally:
        conn.close()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL DEFAULT 'Person',
            gender      TEXT DEFAULT '',
            clothing    TEXT DEFAULT '',
            ai_description TEXT DEFAULT '',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER REFERENCES persons(id),
            type        TEXT NOT NULL CHECK(type IN ('entered','exited')),
            timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
            screenshot  TEXT DEFAULT '',
            crop        TEXT DEFAULT '',
            ai_description TEXT DEFAULT '',
            track_oid   INTEGER DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS face_images (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER NOT NULL REFERENCES persons(id),
            filename    TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_events_person  ON events(person_id);
        CREATE INDEX IF NOT EXISTS idx_events_time    ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_faces_person   ON face_images(person_id);

        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


class PersonDB:
    def __init__(self):
        init_db()

    # --- face / avatar helpers ---

    def _save_img(self, img: np.ndarray, prefix: str) -> str:
        os.makedirs(FACE_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{prefix}_{ts}.jpg"
        path = os.path.join(FACE_DIR, fname)
        try:
            cv2.imwrite(path, img)
        except Exception:
            return ""
        return fname

    # --- core API (called from pipeline) ---

    def save_entry_event(self, track_oid: int, frame_crop: np.ndarray,
                         full_frame: np.ndarray, screenshot_fname: str) -> int:
        """Save entry event with person_id=NULL. Returns event_id."""
        crop_fname = self._save_img(frame_crop, f"crop_oid{track_oid}")
        conn = get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO events (type, screenshot, crop, track_oid)
                   VALUES ('entered', ?, ?, ?)""",
                (screenshot_fname, crop_fname, track_oid),
            )
            event_id = cur.lastrowid
            conn.commit()
            return event_id
        finally:
            conn.close()

    def save_exit_event(self, track_oid: int, person_id: int | None,
                        screenshot_fname: str):
        conn = get_conn()
        try:
            conn.execute(
                """INSERT INTO events (person_id, type, screenshot, track_oid)
                   VALUES (?, 'exited', ?, ?)""",
                (person_id, screenshot_fname, track_oid),
            )
            conn.commit()
        finally:
            conn.close()

    def assign_event(self, event_id: int, person_id: int):
        conn = get_conn()
        try:
            conn.execute("UPDATE events SET person_id = ? WHERE id = ?", (person_id, event_id))
            conn.commit()
        finally:
            conn.close()

    def resolve_event(self, event_id: int, ai_description: str) -> int:
        """Create a new person and assign the event. Returns person_id."""
        conn = get_conn()
        try:
            event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
            if event is None:
                return 0
            if event["person_id"] is not None:
                return event["person_id"]

            cur = conn.execute(
                "INSERT INTO persons (name, ai_description) VALUES (?, ?)",
                (f"Person", f"Person — {ai_description[:200]}" if ai_description else ""),
            )
            pid = cur.lastrowid
            conn.execute("UPDATE persons SET name = ? WHERE id = ?", (f"Person {pid}", pid))
            conn.execute("UPDATE events SET person_id = ?, ai_description = ? WHERE id = ?",
                         (pid, ai_description, event_id))
            conn.commit()
            return pid
        finally:
            conn.close()

    def _match_by_description(self, desc: str, conn: sqlite3.Connection) -> int | None:
        persons = conn.execute("SELECT id, ai_description FROM persons WHERE ai_description != ''").fetchall()
        best_pid = None
        best_score = 0.0
        words1 = set(w.lower() for w in desc.split() if len(w) > 3)
        if not words1:
            return None
        for row in persons:
            words2 = set(w.lower() for w in row["ai_description"].split() if len(w) > 3)
            if not words2:
                continue
            score = len(words1 & words2) / max(len(words1), len(words2))
            if score > best_score:
                best_score = score
                best_pid = row["id"]
        return best_pid if best_score >= 0.55 else None

    def add_face_image(self, person_id: int, img: np.ndarray, suffix: str = ""):
        fname = self._save_img(img, f"person_{person_id}_{suffix}" if suffix else f"person_{person_id}")
        if not fname:
            return
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO face_images (person_id, filename) VALUES (?, ?)",
                (person_id, fname),
            )
            conn.commit()
        finally:
            conn.close()

    # --- face embedding matching (fast, called during save_entry_event if model available) ---

    def get_all_persons_with_descriptions(self) -> list[dict]:
        conn = get_conn()
        try:
            rows = conn.execute("SELECT id, name, ai_description FROM persons ORDER BY id").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def create_person_and_assign(self, event_id: int, ai_description: str) -> int:
        conn = get_conn()
        try:
            event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
            if event is None or event["person_id"] is not None:
                conn.close()
                return event["person_id"] if event else 0

            cur = conn.execute(
                "INSERT INTO persons (name, ai_description) VALUES (?, ?)",
                (f"Person", f"Person — {ai_description[:200]}" if ai_description else ""),
            )
            pid = cur.lastrowid
            conn.execute("UPDATE persons SET name = ? WHERE id = ?", (f"Person {pid}", pid))
            conn.execute("UPDATE events SET person_id = ?, ai_description = ? WHERE id = ?",
                         (pid, ai_description, event_id))
            conn.commit()
            return pid
        finally:
            conn.close()

    # --- person queries (called from server) ---

    def get_person(self, person_id: int) -> dict | None:
        conn = get_conn()
        try:
            row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
            if row is None:
                return None
            face_imgs = [
                r["filename"] for r in conn.execute(
                    "SELECT filename FROM face_images WHERE person_id = ? ORDER BY id",
                    (person_id,),
                ).fetchall()
            ]
            events = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM events WHERE person_id = ? ORDER BY timestamp DESC",
                    (person_id,),
                ).fetchall()
            ]
            ent = sum(1 for e in events if e["type"] == "entered")
            ext = sum(1 for e in events if e["type"] == "exited")
            return {
                "id": row["id"],
                "name": row["name"],
                "gender": row["gender"] or "",
                "clothing": row["clothing"] or "",
                "ai_description": row["ai_description"] or "",
                "entries": ent,
                "exits": ext,
                "face_images": face_imgs,
                "events": events,
            }
        finally:
            conn.close()

    def get_all_persons(self) -> list[dict]:
        conn = get_conn()
        try:
            rows = conn.execute("SELECT * FROM persons ORDER BY id").fetchall()
            result = []
            for row in rows:
                face_imgs = [
                    r["filename"] for r in conn.execute(
                        "SELECT filename FROM face_images WHERE person_id = ? ORDER BY id",
                        (row["id"],),
                    ).fetchall()
                ]
                ent = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE person_id = ? AND type='entered'",
                    (row["id"],),
                ).fetchone()[0]
                ext = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE person_id = ? AND type='exited'",
                    (row["id"],),
                ).fetchone()[0]
                result.append({
                    "id": row["id"],
                    "name": row["name"],
                    "entries": ent,
                    "exits": ext,
                    "face_images": face_imgs,
                    "ai_description": row["ai_description"] or "",
                })
            return result
        finally:
            conn.close()

    def rename(self, person_id: int, new_name: str):
        conn = get_conn()
        try:
            conn.execute("UPDATE persons SET name = ? WHERE id = ?", (new_name, person_id))
            conn.commit()
        finally:
            conn.close()

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        conn = get_conn()
        try:
            rows = conn.execute("""
                SELECT e.*, p.name as person_name
                FROM events e LEFT JOIN persons p ON e.person_id = p.id
                ORDER BY e.timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
