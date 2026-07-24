import sqlite3
import os
import time
import numpy as np
import cv2

DB_PATH = os.path.join(os.path.dirname(__file__), "data.db")
FACE_DIR = os.path.join(os.path.dirname(__file__), "logs", "faces")


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

        CREATE TABLE IF NOT EXISTS face_embeddings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER NOT NULL REFERENCES persons(id),
            embedding   BLOB NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_events_person  ON events(person_id);
        CREATE INDEX IF NOT EXISTS idx_events_time    ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_faces_person   ON face_images(person_id);
        CREATE INDEX IF NOT EXISTS idx_emb_person     ON face_embeddings(person_id);
    """)
    conn.commit()
    conn.close()


class PersonDB:
    def __init__(self):
        init_db()
        self._face_model = None

    # --- face model helpers ---

    def _get_model(self):
        if self._face_model is None:
            try:
                import insightface
                self._face_model = insightface.app.FaceAnalysis(
                    name="buffalo_sc", providers=["CPUExecutionProvider"]
                )
                self._face_model.prepare(ctx_id=-1)
            except Exception:
                self._face_model = False
        return self._face_model if self._face_model is not False else None

    def _extract_face(self, img: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        model = self._get_model()
        if model is None:
            return None, None
        try:
            faces = model.get(img)
        except Exception:
            faces = []
        if faces:
            x1, y1, x2, y2 = [int(v) for v in faces[0].bbox]
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                face_crop = img[y1:y2, x1:x2]
                return face_crop, faces[0].normed_embedding
        return None, None

    def _upper_body_crop(self, img: np.ndarray) -> np.ndarray:
        h = img.shape[0]
        return img[:max(1, int(h * 0.35)), :]

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

    def resolve_event(self, event_id: int, ai_description: str):
        """Background: match description against known persons, assign person_id."""
        if not ai_description:
            return
        conn = get_conn()
        try:
            event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
            if event is None or event["person_id"] is not None:
                return

            matched_pid = self._match_by_description(ai_description, conn)
            if matched_pid is None:
                conn.execute(
                    "INSERT INTO persons (name, ai_description) VALUES (?, ?)",
                    (f"Person", f"Person — {ai_description[:100]}"),
                )
                matched_pid = conn.lastrowid
                conn.execute("UPDATE persons SET name = ? WHERE id = ?",
                             (f"Person {matched_pid}", matched_pid))

            conn.execute("UPDATE events SET person_id = ?, ai_description = ? WHERE id = ?",
                         (matched_pid, ai_description, event_id))
            conn.commit()
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

    def try_match_by_face(self, crop: np.ndarray) -> int | None:
        """Try face embedding matching. Returns person_id or None."""
        face_crop, face_emb = self._extract_face(crop)
        if face_emb is None:
            return None
        conn = get_conn()
        try:
            rows = conn.execute("SELECT person_id, embedding FROM face_embeddings").fetchall()
            for row in rows:
                known = np.frombuffer(row["embedding"], dtype=np.float32)
                if float(face_emb @ known) >= 0.35:
                    return row["person_id"]
        finally:
            conn.close()

        # No match but have face → save embedding
        conn2 = get_conn()
        try:
            conn2.execute(
                "INSERT INTO face_embeddings (person_id, embedding) VALUES (?, ?)",
                (None, face_emb.tobytes()),
            )
            conn2.commit()
        finally:
            conn2.close()
        return None

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
