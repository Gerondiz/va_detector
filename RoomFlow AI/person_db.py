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
                dc_list = info.get("dominant_colors", [])
                info["dominant_colors"] = [[tuple(c) for c in group] for group in dc_list]
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
            entry["dominant_colors"] = [[list(c) for c in group] for group in entry.get("dominant_colors", [])]
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

    def _match_by_hist(self, body_img: np.ndarray) -> tuple[int | None, float]:
        hist = self._extract_color_hist(body_img)
        if hist is None:
            return None, 0.0
        best_id = None
        best_score = -1.0
        for pid, info in self.people.items():
            for known_hist in info.get("color_hist", []):
                score = self._hist_similarity(hist, known_hist)
                if score > best_score:
                    best_score = score
                    best_id = pid
        return best_id, best_score

    @staticmethod
    def _extract_dominant_colors(img: np.ndarray, k: int = 3) -> list[tuple[int, int, int]]:
        if img is None or img.size == 0:
            return []
        try:
            h, w = img.shape[:2]
            pixels = img.reshape(-1, 3)
            if len(pixels) > 1000:
                idx = np.random.choice(len(pixels), 1000, replace=False)
                pixels = pixels[idx]
            pixels = np.float32(pixels)
            _, labels, centers = cv2.kmeans(pixels, k, None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
            counts = np.bincount(labels.flatten())
            ordered = [tuple(map(int, centers[i])) for i in np.argsort(-counts)]
            return ordered
        except Exception:
            return []

    @staticmethod
    def _color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

    def _match_by_color(self, body_img: np.ndarray, threshold: float = 80.0) -> int | None:
        colors = self._extract_dominant_colors(body_img)
        if not colors:
            return None
        best_id = None
        best_score = float("inf")
        for pid, info in self.people.items():
            for known_colors in info.get("dominant_colors", []):
                total = sum(min(self._color_distance(c, kc) for kc in known_colors) for c in colors)
                avg_dist = total / len(colors)
                if avg_dist < best_score:
                    best_score = avg_dist
                    best_id = pid
        if best_id is not None and best_score < threshold:
            return best_id
        return None

    @staticmethod
    def _description_similarity(desc1: str, desc2: str) -> float:
        if not desc1 or not desc2:
            return 0.0
        words1 = set(w.lower() for w in desc1.split() if len(w) > 3)
        words2 = set(w.lower() for w in desc2.split() if len(w) > 3)
        if not words1 or not words2:
            return 0.0
        intersect = words1 & words2
        return len(intersect) / max(len(words1), len(words2))

    def _match_by_description(self, ai_desc: str, threshold: float = 0.55) -> int | None:
        if not ai_desc:
            return None
        best_id = None
        best_score = 0.0
        for pid, info in self.people.items():
            score = self._description_similarity(ai_desc, info.get("ai_description", ""))
            if score > best_score:
                best_score = score
                best_id = pid
        return best_id if best_score >= threshold else None

    def resolve_with_ai(self, pid: int, ai_description: str) -> int:
        """Called after AI description arrives. May merge pid into an existing person."""
        if pid not in self.people:
            return pid
        if not ai_description:
            return pid
        existing = self._match_by_description(ai_description)
        if existing is not None and existing != pid:
            self._merge_persons(pid, existing)
            return existing
        self.people[pid]["ai_description"] = ai_description
        self._save()
        return pid

    def _merge_persons(self, src_pid: int, dst_pid: int):
        if src_pid not in self.people or dst_pid not in self.people:
            return
        src = self.people[src_pid]
        dst = self.people[dst_pid]
        dst["entries"] += src.get("entries", 0)
        dst["exits"] += src.get("exits", 0)
        dst["face_images"].extend(src.get("face_images", []))
        dst["face_images"] = dst["face_images"][-10:]
        dst["embeddings"].extend(src.get("embeddings", []))
        dst["embeddings"] = dst["embeddings"][-5:]
        dst["color_hist"].extend(src.get("color_hist", []))
        dst["color_hist"] = dst["color_hist"][-3:]
        dst["last_seen"] = max(dst.get("last_seen", ""), src.get("last_seen", ""))
        del self.people[src_pid]
        self._save()

    def _extract_face(self, body_img: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Try to detect face inside body_img. Returns (face_crop, face_embedding) or (None, None)."""
        model = self._get_model()
        if model is None:
            return None, None
        try:
            faces = model.get(body_img)
        except Exception:
            faces = []
        if faces:
            x1, y1, x2, y2 = [int(v) for v in faces[0].bbox]
            h, w = body_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                face_crop = body_img[y1:y2, x1:x2]
                return face_crop, faces[0].normed_embedding
        return None, None

    def _upper_body_crop(self, img: np.ndarray) -> np.ndarray:
        """Take upper 35% of the image (head/shoulders area)."""
        h = img.shape[0]
        return img[:max(1, int(h * 0.35)), :]

    def identify(self, body_crop: np.ndarray, track_id: int) -> int:
        face_crop, face_emb = self._extract_face(body_crop)
        display_img = face_crop if face_crop is not None else self._upper_body_crop(body_crop)

        if face_emb is not None:
            best_id, best_score = None, -1
            for pid, info in self.people.items():
                for known_emb in info.get("embeddings", []):
                    score = float(face_emb @ known_emb)
                    if score > best_score:
                        best_score = score
                        best_id = pid
            if best_id is not None and best_score >= 0.35:
                return self._update_person(best_id, display_img, face_emb, body_crop)

        hist_id, hist_score = self._match_by_hist(body_crop)
        if hist_id is not None and hist_score >= 0.5:
            return self._update_person(hist_id, display_img, face_emb, body_crop)

        color_id = self._match_by_color(body_crop)
        if color_id is not None:
            return self._update_person(color_id, display_img, face_emb, body_crop)

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
            "dominant_colors": [],
            "ai_description": "",
        }
        self._save_face(pid, display_img, "first")
        self._update_hist(pid, body_crop)
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
        self._update_dominant_colors(pid, body_img)
        self._save()
        return pid

    def _update_dominant_colors(self, pid: int, body_img: np.ndarray):
        colors = self._extract_dominant_colors(body_img)
        if colors:
            info = self.people[pid]
            if "dominant_colors" not in info:
                info["dominant_colors"] = []
            if not info["dominant_colors"]:
                info["dominant_colors"].append(colors)
            else:
                last = info["dominant_colors"][-1]
                avg_dist = sum(min(self._color_distance(c, lc) for lc in last) for c in colors) / len(colors)
                if avg_dist > 40:
                    info["dominant_colors"].append(colors)
                    if len(info["dominant_colors"]) > 3:
                        info["dominant_colors"] = info["dominant_colors"][-3:]

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
            "dominant_colors": [],
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
                fname = os.path.basename(path)
                self.people[person_id]["face_images"].append(fname)
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
