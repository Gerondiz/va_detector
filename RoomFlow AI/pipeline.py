import cv2
import time
import os
import json
import queue
import threading
import traceback
import numpy as np
from dataclasses import dataclass, field
from config import CAMERA, LOGS, MODELS
from detector import Detector
from llm_client import LLMAnalyzer
from logger import EventLogger
from person_db import PersonDB

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")



def _load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(data: dict):
    existing = _load_settings()
    existing.update(data)
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, SETTINGS_FILE)


def _settings_defaults():
    s = _load_settings()
    s.setdefault("door_left", 0.3)
    s.setdefault("door_right", 0.7)
    return s

@dataclass
class SharedState:
    frame: np.ndarray | None = None
    entered: int = 0
    exited: int = 0
    current_people: int = 0
    person_visits: dict = field(default_factory=dict)
    person_events: list = field(default_factory=list)
    recent_events: list = field(default_factory=list)
    running: bool = True
    line_position: float = field(default_factory=lambda: _load_settings().get("line_position", 0.7))
    door_left: float = 0.3
    door_right: float = 0.7
    door_top: float = 0.0
    door_bottom: float = 1.0
    fps: int = 0
    person_db: PersonDB | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        s = _load_settings()
        self.entered = s.get("entered", 0)
        self.exited = s.get("exited", 0)
        self.door_left = s.get("door_left", 0.3)
        self.door_right = s.get("door_right", 0.7)
        self.door_top = s.get("door_top", 0.0)
        self.door_bottom = s.get("door_bottom", 1.0)

    def update_frame(self, f: np.ndarray):
        with self._lock:
            self.frame = f.copy()

    def get_frame(self):
        with self._lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def add_event(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.recent_events.append(entry)
        if len(self.recent_events) > 100:
            self.recent_events = self.recent_events[-100:]

    def get_events(self, n: int = 20):
        return self.recent_events[-n:]

    def add_person_event(self, event: dict):
        self.person_events.append(event)

    def get_person_events(self, n: int = 50):
        return self.person_events[-n:]

    def set_line_position(self, value: float):
        self.line_position = max(0.0, min(1.0, value))
        _save_settings({"line_position": self.line_position})

    def set_door_zone(self, left: float, right: float, top: float = 0.0, bottom: float = 1.0):
        self.door_left = max(0.0, min(1.0, left))
        self.door_right = max(0.0, min(1.0, right))
        self.door_top = max(0.0, min(1.0, top))
        self.door_bottom = max(0.0, min(1.0, bottom))
        if self.door_left >= self.door_right:
            self.door_right = min(1.0, self.door_left + 0.1)
        if self.door_top >= self.door_bottom:
            self.door_bottom = min(1.0, self.door_top + 0.1)
        _save_settings({"door_left": self.door_left, "door_right": self.door_right,
                         "door_top": self.door_top, "door_bottom": self.door_bottom})





class PeopleCounter:
    def __init__(self, state: SharedState, logger: EventLogger, person_db=None, llm=None):
        self.state = state
        self.logger = logger
        self.person_db = person_db
        self.llm = llm
        self.tracked: dict[int, dict] = {}
        self.cooldown: dict[int, float] = {}
        self._pos_cooldown: list[tuple[int, int, float]] = []
        self._recently_lost: dict[int, dict] = {}
        self._lost_cleanup_counter = 0
        self._ai_queue: queue.Queue = queue.Queue()
        if llm is not None:
            t = threading.Thread(target=self._ai_worker, daemon=True)
            t.start()

    def _match_lost_track(self, oid: int, x1, y1, x2, y2) -> int | None:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        best = None
        best_dist = 300
        for lost_id, lost_info in self._recently_lost.items():
            lx1, ly1, lx2, ly2 = lost_info.get("lost_bbox", (0, 0, 0, 0))
            lcx = (lx1 + lx2) / 2
            lcy = (ly1 + ly2) / 2
            dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = lost_id
        return best

    def update(self, objects: list[dict], frame: np.ndarray, annotated: np.ndarray):
        h, w = frame.shape[:2]
        door_x1 = int(w * self.state.door_left)
        door_x2 = int(w * self.state.door_right)
        door_y1 = int(h * self.state.door_top)
        door_y2 = int(h * self.state.door_bottom)

        persons = [o for o in objects if o["label"] == "person" and o["id"] is not None]
        current_ids = {o["id"] for o in persons}
        now = time.time()

        for o in persons:
            oid = o["id"]
            x1, y1, x2, y2 = o["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            in_door = door_x1 <= cx <= door_x2 and door_y1 <= cy <= door_y2

            if oid not in self.tracked:
                matched = self._match_lost_track(oid, x1, y1, x2, y2)
                if matched is not None:
                    self.tracked[oid] = self._recently_lost.pop(matched)
                    self.tracked[oid]["prev_in_door"] = self.tracked[oid]["in_door"]
                    self.tracked[oid]["in_door"] = in_door
                    if matched in self.tracked:
                        del self.tracked[matched]
                else:
                    self.tracked[oid] = {"prev_in_door": in_door, "in_door": in_door,
                                          "exit_frame": None, "person_id": None,
                                          "ever_in_door": in_door,
                                          "entered_from_outside": False}
                    if in_door and self.person_db is not None:
                        self._identify_person(oid, frame, x1, y1, x2, y2)
                d = self.tracked[oid]
            else:
                d = self.tracked[oid]
                was_in_door = d["in_door"]
                if in_door:
                    d["ever_in_door"] = True
                    if not was_in_door:
                        d["entered_from_outside"] = True
                        d["exit_frame"] = annotated.copy()
                        if self.person_db is not None and d.get("person_id") is None:
                            self._identify_person(oid, frame, x1, y1, x2, y2)
                if was_in_door and not in_door and d.get("entered_from_outside"):
                    self._pos_cooldown = [(px, py, pt) for (px, py, pt) in self._pos_cooldown if now - pt < 10]
                    too_soon = any(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 < 100 for (px, py, pt) in self._pos_cooldown)
                    if not too_soon:
                        self._pos_cooldown.append((int(cx), int(cy), now))
                        self._count(oid, "entered", frame, annotated, self.cooldown)
                d["prev_in_door"] = was_in_door
                d["in_door"] = in_door

            d["_last_bbox_x1"] = x1
            d["_last_bbox_y1"] = y1
            d["_last_bbox_x2"] = x2
            d["_last_bbox_y2"] = y2

            label = self._person_label(oid)
            cv2.circle(annotated, (int(cx), int(y2)), 4, (0, 255, 255), -1)
            cv2.putText(annotated, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        for oid in list(self.tracked.keys()):
            if oid not in current_ids:
                d = self.tracked[oid]
                self._recently_lost[oid] = d
                self._recently_lost[oid]["lost_time"] = now
                self._recently_lost[oid]["lost_bbox"] = (
                    d.get("_last_bbox_x1", 0),
                    d.get("_last_bbox_y1", 0),
                    d.get("_last_bbox_x2", 0),
                    d.get("_last_bbox_y2", 0),
                )
                in_door_now = d.get("in_door", False)
                in_door_prev = d.get("prev_in_door", False)
                if in_door_now or in_door_prev:
                    exit_img = d.get("exit_frame")
                    if exit_img is None:
                        exit_img = annotated
                    self._recently_lost[oid]["pending_exit"] = True
                    self._recently_lost[oid]["exit_img"] = exit_img
                else:
                    self._recently_lost[oid]["pending_exit"] = False
                del self.tracked[oid]

        for lost_id in list(self._recently_lost.keys()):
            info = self._recently_lost[lost_id]
            elapsed = now - info.get("lost_time", 0)
            if info.get("pending_exit") and elapsed > 0.5:
                info["pending_exit"] = False
                exit_img = info.get("exit_img")
                if exit_img is None:
                    exit_img = annotated
                self._count(lost_id, "exited", exit_img, exit_img, self.cooldown)
            if elapsed > 5.0:
                del self._recently_lost[lost_id]

        self.state.current_people = 0

        cv2.rectangle(annotated, (door_x1, door_y1), (door_x2, door_y2), (255, 255, 0), 2)
        cv2.putText(annotated, "DOOR", (door_x1 + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(annotated, f"IN: {self.state.entered}  OUT: {self.state.exited}  NOW: {self.state.current_people}",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    def _person_label(self, oid: int) -> str:
        d = self.tracked.get(oid)
        if d is None:
            return f"#{oid}"
        pid = d.get("person_id")
        if pid is not None and self.person_db is not None:
            info = self.person_db.people.get(pid)
            if info:
                return info.get("name", f"#{oid}")
        return f"#{oid}"

    def _identify_person(self, oid: int, frame: np.ndarray, x1, y1, x2, y2):
        try:
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            crop = frame[max(0, y1i):min(frame.shape[0], y2i),
                         max(0, x1i):min(frame.shape[1], x2i)]
            if crop.size == 0 or self.person_db is None:
                return
            pid = self.person_db.identify(crop, oid)
            self.tracked[oid]["person_id"] = pid
            if self.llm is not None:
                self._ai_queue.put((pid, crop))
        except Exception as e:
            with open("logs/pipeline_error.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} _identify_person: {e}\n")
                traceback.print_exc(file=f)

    def _ai_worker(self):
        while self.state.running:
            try:
                pid, crop = self._ai_queue.get(timeout=1)
                if crop is None or crop.size == 0:
                    continue
                desc = self.llm.describe_person(crop)
                if desc and "error" not in desc.lower():
                    resolved = self.person_db.resolve_with_ai(pid, desc)
                    if resolved != pid:
                        for d in self.tracked.values():
                            if d.get("person_id") == pid:
                                d["person_id"] = resolved
            except queue.Empty:
                continue
            except Exception as e:
                with open("logs/pipeline_error.log", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} _ai_worker: {e}\n")

    def _person_name(self, oid: int) -> str:
        d = self.tracked.get(oid)
        if d is None:
            return f"#{oid}"
        pid = d.get("person_id")
        if pid is not None and self.person_db is not None:
            info = self.person_db.people.get(pid)
            if info:
                return info.get("name", f"#{oid}")
        return f"#{oid}"

    def _count(self, oid: int, direction: str, frame: np.ndarray, annotated: np.ndarray, cooldown: dict = None):
        now = time.time()
        cd = cooldown if cooldown is not None else {}
        if oid in cd and now - cd[oid] < 3:
            return
        cd[oid] = now

        if direction == "entered":
            self.state.entered += 1
        else:
            self.state.exited += 1
            pid = self.tracked.get(oid, {}).get("person_id")
            if pid is not None and self.person_db is not None:
                try:
                    self.person_db.record_exit(pid)
                except Exception:
                    pass

        _save_settings({"line_position": self.state.line_position,
                        "door_left": self.state.door_left,
                        "door_right": self.state.door_right,
                        "entered": self.state.entered, "exited": self.state.exited})

        pname = self._person_name(oid)
        self.state.add_event(f"{direction} {pname} (IN: {self.state.entered} OUT: {self.state.exited})")
        self.logger.log(f"person_{direction}", {"id": oid, "person": pname})
        if self.llm is not None:
            try:
                desc = self.llm.describe_person(frame)
                self.state.add_event(f"[vision] {pname} {direction}: {desc}")
            except Exception:
                pass
        self._save_crossing(oid, direction, frame, annotated)
        self._save_person_event(oid, direction)

    def _save_crossing(self, oid: int, direction: str, frame: np.ndarray, annotated: np.ndarray):
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"logs/person_{oid}_{direction}_{ts}.jpg"
        cv2.imwrite(path, annotated)

    def _save_person_event(self, oid: int, direction: str):
        visits = self.state.person_visits
        visits[oid] = visits.get(oid, 0) + 1
        self.state.add_person_event({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "person_id": oid,
            "direction": direction,
            "total_visits": visits[oid],
        })


class CameraPipeline(threading.Thread):
    def __init__(self, state: SharedState):
        super().__init__(daemon=True)
        self.state = state
        self.logger = EventLogger()

    def _connect(self) -> cv2.VideoCapture | None:
        url = CAMERA["mjpg_url"] if CAMERA["use_mjpg"] else CAMERA["rtsp_url"]
        cap = cv2.VideoCapture()
        if not CAMERA["use_mjpg"]:
            cap.set(111, 0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.open(url)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def run(self):
        cap = self._connect()
        if cap is None:
            print("[ERROR] Cannot open camera")
            self.state.running = False
            return

        detector = Detector()
        if self.state.person_db is None:
            self.state.person_db = PersonDB()
        llm = LLMAnalyzer()
        counter = PeopleCounter(self.state, self.logger, person_db=self.state.person_db, llm=llm)

        frame_count = 0
        fps_start = time.time()
        track_reset = 0

        while self.state.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Lost frame, reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = self._connect()
                    if cap is None:
                        time.sleep(2)
                    continue

                frame_count += 1
                if frame_count % 2 == 0:
                    track_reset += 1
                    if track_reset >= 1500:
                        detector = Detector()
                        track_reset = 0

                    results = detector.detect(frame, track=True)
                    objects = detector.get_objects(results)
                    annotated = detector.get_annotated_frame(results)

                    counter.update(objects, frame, annotated)

                    # LLM text analysis disabled (gemma4 text-only, no vision)

                    cv2.putText(annotated, f"FPS: {self.state.fps}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    self.state.update_frame(annotated)
                else:
                    self.state.update_frame(frame)

                if time.time() - fps_start >= 1.0:
                    self.state.fps = frame_count
                    frame_count = 0
                    fps_start = time.time()

            except Exception as e:
                with open("logs/pipeline_error.log", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {e}\n")
                    traceback.print_exc(file=f)
                print(f"[ERROR] Pipeline: {e} — see logs/pipeline_error.log")
                time.sleep(2)

        cap.release()
