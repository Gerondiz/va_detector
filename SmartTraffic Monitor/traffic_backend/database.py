import sqlite3
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "traffic_data.db")

SETTINGS_CACHE: dict[str, str] = {}
_SETTINGS_DB_INIT = False

COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}


def _ensure_settings():
    global _SETTINGS_DB_INIT
    if _SETTINGS_DB_INIT:
        return
    conn = get_conn()
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.commit()
        _SETTINGS_DB_INIT = True
    finally:
        conn.close()


def get_setting(key: str, default: str = "") -> str:
    _ensure_settings()
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
    _ensure_settings()
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
    return conn


class TrafficDB:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        conn = get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_class TEXT NOT NULL,
                direction     TEXT NOT NULL CHECK(direction IN ('left','right')),
                timestamp     TEXT NOT NULL DEFAULT (datetime('now')),
                screenshot    TEXT DEFAULT '',
                crop          TEXT DEFAULT '',
                confidence    REAL DEFAULT 0.0,
                track_id      INTEGER DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS daily_stats (
                date        TEXT PRIMARY KEY,
                car_left    INTEGER DEFAULT 0,
                car_right   INTEGER DEFAULT 0,
                bus_left    INTEGER DEFAULT 0,
                bus_right   INTEGER DEFAULT 0,
                truck_left  INTEGER DEFAULT 0,
                truck_right INTEGER DEFAULT 0,
                moto_left   INTEGER DEFAULT 0,
                moto_right  INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_class ON events(vehicle_class);
        """)
        conn.commit()
        conn.close()

    def save_pass_event(self, vehicle_class: str, direction: str,
                        confidence: float, track_id: int) -> int:
        conn = get_conn()
        try:
            today = time.strftime("%Y-%m-%d")
            cur = conn.execute(
                """INSERT INTO events (vehicle_class, direction, confidence, track_id)
                   VALUES (?, ?, ?, ?)""",
                (vehicle_class, direction, confidence, track_id),
            )
            eid = cur.lastrowid
            # upsert daily_stats
            col = f"{vehicle_class}_{direction}"
            conn.execute(f"""
                INSERT INTO daily_stats (date, {col}) VALUES (?, 1)
                ON CONFLICT(date) DO UPDATE SET {col} = {col} + 1
            """, (today,))
            conn.commit()
            return eid
        finally:
            conn.close()

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self, days: int = 7) -> list[dict]:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM daily_stats ORDER BY date DESC LIMIT ?", (days,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_totals(self) -> dict:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT vehicle_class, direction, COUNT(*) as cnt FROM events GROUP BY vehicle_class, direction"
            ).fetchall()
            totals = {}
            for r in rows:
                key = f"{r['vehicle_class']}_{r['direction']}"
                totals[key] = r["cnt"]
            return totals
        finally:
            conn.close()

    def clear_events(self):
        conn = get_conn()
        try:
            conn.execute("DELETE FROM events")
            conn.execute("DELETE FROM daily_stats")
            conn.commit()
        finally:
            conn.close()
