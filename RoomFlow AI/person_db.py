import json
import os
import time
import numpy as np
import cv2

DB_FILE = os.path.join(os.path.dirname(__file__), "person_db.json")
FACE_DIR = os.path.join(os.path.dirname(__file__), "logs", "faces")


class PersonDB:
    def __init__(self):
        self.next_id: int = 1
        self.people: dict[int, dict] = {}
        self._load()
        self._face_model = None

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

    def _load(self):
        try:
            with open(DB_FILE) as f:
                data = json.load(f)
            self.next_id = data.get("next_id", 1)
            raw = data.get("people", {})
            for pid_str, info in raw.items():
                pid = int(pid_str)
                emb_list = info.get("embeddings", [])
                info["embeddings"] = [np.array(e, dtype=np.float32) for e in emb_list]
                hist_list = info.get("color_hist", [])
                info["color_hist"] = [np.array(h, dtype=np.float32) for h in hist_list]
                self.people[pid] = info
        except (FileNotFoundError, json.JSONDecodeError):
            self.people = {}
            self.next_id = 1

    def _save(self):
        data = {
            "next_id": self.next_id,
            "people": {},
        }
        for pid, info in self.people.items():
            entry = dict(info)
            entry["embeddings"] = [e.tolist() for e in entry.get("embeddings", [])]
            entry["color_hist"] = [h.tolist() for h in entry.get("color_hist", [])]
            data["people"][str(pid)] = entry
        with open(DB_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _extract_color_hist(img: np.ndarray) -> np.ndarray | None:
        if img is None or img.size == 0:
            return None
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist.astype(np.float32).flatten()
        except Exception:
            return None

    @staticmethod
    def _hist_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
        return float(cv2.compareHist(h1.reshape(30, 32), h2.reshape(30, 32), cv2.HISTCMP_CORREL))

    def _match_by_hist(self, body_img: np.ndarray) -> int | None:
        hist = self._extract_color_hist(body_img)
        if hist is None:
            return None
        best_id = None
        best_score = -1
        for pid, info in self.people.items():
            for known_hist in info.get("color_hist", []):
                score = self._hist_similarity(hist, known_hist)
                if score > best_score:
                    best_score = score
                    best_id = pid
        if best_id is not None and best_score > 0.85:
            return best_id
        return None

    def identify(self, face_img: np.ndarray, full_body_img: np.ndarray, track_id: int) -> int:
        model = self._get_model()
        face_emb = None
        if model is not None:
            try:
                faces = model.get(face_img)
            except Exception:
                faces = []
            if faces:
                face_emb = faces[0].normed_embedding

        best_id = None
        best_score = -1

        if face_emb is not None:
            for pid, info in self.people.items():
                for known_emb in info.get("embeddings", []):
                    score = float(face_emb @ known_emb)
                    if score > best_score:
                        best_score = score
                        best_id = pid
            if best_id is not None and best_score >= 0.35:
                return self._update_person(best_id, face_img, face_emb, full_body_img)

        matched_by_hist = self._match_by_hist(full_body_img)
        if matched_by_hist is not None:
            return self._update_person(matched_by_hist, face_img, face_emb, full_body_img)

        pid = self.next_id
        self.next_id += 1
        name = f"Person {pid}"
        self.people[pid] = {
            "name": name,
            "embeddings": [],
            "entries": 1,
            "exits": 0,
            "last_seen": time.strftime("%Y-%m-%d %H:%M:%S"),
            "face_images": [],
            "color_hist": [],
            "ai_description": "",
        }
        self._save_face(pid, face_img, "first")
        self._update_hist(pid, full_body_img)
        if face_emb is not None:
            self.people[pid]["embeddings"] = [face_emb]
        self._save()
        return pid

    def _update_person(self, pid: int, face_img: np.ndarray, face_emb, body_img: np.ndarray) -> int:
        info = self.people[pid]
        info["entries"] = info.get("entries", 0) + 1
        info["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if face_emb is not None:
            matched = False
            for known_emb in info.get("embeddings", []):
                if float(face_emb @ known_emb) > 0.9:
                    matched = True
                    break
            if not matched:
                info["embeddings"].append(face_emb)
                if len(info["embeddings"]) > 5:
                    info["embeddings"] = info["embeddings"][-5:]
        self._update_hist(pid, body_img)
        self._save()
        return pid

    def _update_hist(self, pid: int, body_img: np.ndarray):
        hist = self._extract_color_hist(body_img)
        if hist is not None:
            info = self.people[pid]
            if "color_hist" not in info:
                info["color_hist"] = []
            if not info["color_hist"] or max(self._hist_similarity(hist, h) for h in info["color_hist"]) < 0.95:
                info["color_hist"].append(hist)
                if len(info["color_hist"]) > 3:
                    info["color_hist"] = info["color_hist"][-3:]

    def _assign_unknown(self, track_id: int) -> int:
        pid = self.next_id
        self.next_id += 1
        name = f"Person {pid}"
        self.people[pid] = {
            "name": name,
            "embeddings": [],
            "entries": 1,
            "exits": 0,
            "last_seen": time.strftime("%Y-%m-%d %H:%M:%S"),
            "face_images": [],
            "color_hist": [],
            "ai_description": "",
        }
        self._save()
        return pid

    def record_exit(self, person_id: int):
        if person_id in self.people:
            self.people[person_id]["exits"] = self.people[person_id].get("exits", 0) + 1
            self.people[person_id]["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save()

    def _save_face(self, person_id: int, img: np.ndarray, suffix: str = ""):
        os.makedirs(FACE_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(FACE_DIR, f"person_{person_id}_{suffix}_{ts}.jpg")
        try:
            cv2.imwrite(path, img)
            if person_id in self.people:
                rel = os.path.relpath(path, os.path.dirname(DB_FILE))
                self.people[person_id]["face_images"].append(rel)
                if len(self.people[person_id]["face_images"]) > 10:
                    self.people[person_id]["face_images"] = self.people[person_id]["face_images"][-10:]
        except Exception:
            pass

    def set_description(self, person_id: int, description: str):
        if person_id in self.people and description:
            self.people[person_id]["ai_description"] = description
            self._save()

    def rename(self, person_id: int, new_name: str):
        if person_id in self.people:
            self.people[person_id]["name"] = new_name
            self._save()

    def get_all(self) -> list[dict]:
        result = []
        for pid in sorted(self.people.keys()):
            info = self.people[pid]
            result.append({
                "id": pid,
                "name": info.get("name", f"Person {pid}"),
                "entries": info.get("entries", 0),
                "exits": info.get("exits", 0),
                "last_seen": info.get("last_seen", ""),
                "face_images": info.get("face_images", []),
                "has_face": len(info.get("embeddings", [])) > 0,
                "ai_description": info.get("ai_description", ""),
            })
        return result

    def to_dict(self) -> dict:
        return {"next_id": self.next_id, "people": self.get_all()}
