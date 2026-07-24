import cv2
import time
import os

import queue
import threading
import traceback
import numpy as np
from dataclasses import dataclass, field
from .config import CAMERA, LOGS, MODELS
from .detector import Detector
from .llm_client import LLMAnalyzer
from .logger import EventLogger
from .database import PersonDB

from .database import get_setting, set_setting

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
    door_left: float = 0.3
    door_right: float = 0.7
    door_top: float = 0.0
    door_bottom: float = 1.0
    fps: int = 0
    person_db: PersonDB | None = None
    last_frame_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.door_left = float(get_setting("door_left", "0.3"))
        self.door_right = float(get_setting("door_right", "0.7"))
        self.door_top = float(get_setting("door_top", "0.0"))
        self.door_bottom = float(get_setting("door_bottom", "1.0"))
        self.entered = int(get_setting("entered", "0"))
        self.exited = int(get_setting("exited", "0"))

    def update_frame(self, f: np.ndarray):
        with self._lock:
            self.frame = f.copy()
            self.last_frame_time = time.time()

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

    def set_door_zone(self, left: float, right: float, top: float = 0.0, bottom: float = 1.0):
        self.door_left = max(0.0, min(1.0, left))
        self.door_right = max(0.0, min(1.0, right))
        self.door_top = max(0.0, min(1.0, top))
        self.door_bottom = max(0.0, min(1.0, bottom))
        if self.door_left >= self.door_right:
            self.door_right = min(1.0, self.door_left + 0.1)
        if self.door_top >= self.door_bottom:
            self.door_bottom = min(1.0, self.door_top + 0.1)
        set_setting("door_left", str(self.door_left))
        set_setting("door_right", str(self.door_right))
        set_setting("door_top", str(self.door_top))
        set_setting("door_bottom", str(self.door_bottom))





class PeopleCounter:
    def __init__(self, state: SharedState, logger: EventLogger, db=None, llm=None):
        self.state = state
        self.logger = logger
        self.db = db
        self.llm = llm
        self.tracked: dict[int, dict] = {}
        self._pos_cooldown: list[tuple[int, int, float]] = []
        self._recently_lost: dict[int, dict] = {}
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

    def _save_crossing(self, oid: int, direction: str, frame: np.ndarray):
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(LOGS["dir"], exist_ok=True)
        path = os.path.join(LOGS["dir"], f"person_{oid}_{direction}_{ts}.jpg")
        success = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            self.state.add_event(f"FAILED to save screenshot {path}")
            return ""
        return f"person_{oid}_{direction}_{ts}.jpg"

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
            in_door = door_x1 <= cx <= door_x2 and door_y1 <= y2 <= door_y2

            if oid not in self.tracked:
                matched = self._match_lost_track(oid, x1, y1, x2, y2)
                if matched is not None:
                    self.tracked[oid] = self._recently_lost.pop(matched)
                    self.tracked[oid]["prev_in_door"] = self.tracked[oid]["in_door"]
                    self.tracked[oid]["in_door"] = in_door
                else:
                    self.tracked[oid] = {
                        "prev_in_door": in_door, "in_door": in_door,
                        "person_id": None, "event_id": None,
                    }
                d = self.tracked[oid]
            else:
                d = self.tracked[oid]
                was_in_door = d["in_door"]

                if was_in_door and not in_door:
                    self._pos_cooldown = [(px, py, pt) for (px, py, pt) in self._pos_cooldown if now - pt < 10]
                    too_soon = any(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 < 100 for (px, py, pt) in self._pos_cooldown)
                    if not too_soon:
                        self._pos_cooldown.append((int(cx), int(cy), now))
                        self.state.entered += 1
                        set_setting("entered", str(self.state.entered))
                        screenshot = self._save_crossing(oid, "entered", annotated)
                        if self.db is not None:
                            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                            crop = frame[max(0, y1i):min(h, y2i), max(0, x1i):min(w, x2i)]
                            eid = self.db.save_entry_event(oid, crop, frame, screenshot)
                            d["event_id"] = eid
                            if self.llm is not None:
                                self._ai_queue.put((eid, crop))
                        self.state.add_event(
                            f"entered #{oid} (IN: {self.state.entered} OUT: {self.state.exited})")

                if not was_in_door and in_door:
                    d["exit_frame"] = annotated.copy()

                d["prev_in_door"] = was_in_door
                d["in_door"] = in_door

            d["_last_bbox_x1"] = x1
            d["_last_bbox_y1"] = y1
            d["_last_bbox_x2"] = x2
            d["_last_bbox_y2"] = y2

            label = f"#{oid}"
            if d.get("person_id"):
                label = f"P{d['person_id']}"
            cv2.circle(annotated, (int(cx), int(y2)), 4, (0, 255, 255), -1)
            cv2.putText(annotated, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        for oid in list(self.tracked.keys()):
            if oid not in current_ids:
                d = self.tracked[oid]
                self._recently_lost[oid] = d
                self._recently_lost[oid]["lost_time"] = now
                self._recently_lost[oid]["lost_bbox"] = (
                    d.get("_last_bbox_x1", 0), d.get("_last_bbox_y1", 0),
                    d.get("_last_bbox_x2", 0), d.get("_last_bbox_y2", 0),
                )
                in_door_now = d.get("in_door", False)
                in_door_prev = d.get("prev_in_door", False)
                if in_door_now or in_door_prev:
                    self._recently_lost[oid]["pending_exit"] = True
                    self._recently_lost[oid]["exit_img"] = d.get("exit_frame") if d.get("exit_frame") is not None else annotated
                    self._recently_lost[oid]["person_id"] = d.get("person_id")
                del self.tracked[oid]

        for lost_id in list(self._recently_lost.keys()):
            info = self._recently_lost[lost_id]
            elapsed = now - info.get("lost_time", 0)
            if info.get("pending_exit") and elapsed > 0.5:
                info["pending_exit"] = False
                self.state.exited += 1
                set_setting("exited", str(self.state.exited))
                exit_img = info.get("exit_img") if info.get("exit_img") is not None else annotated
                screenshot = self._save_crossing(lost_id, "exited", exit_img)
                pid = info.get("person_id")
                if self.db is not None:
                    self.db.save_exit_event(lost_id, pid, screenshot)
                self.state.add_event(
                    f"exited #{lost_id} (IN: {self.state.entered} OUT: {self.state.exited})")
            if elapsed > 5.0:
                del self._recently_lost[lost_id]

        self.state.current_people = 0

        cv2.rectangle(annotated, (door_x1, door_y1), (door_x2, door_y2), (255, 255, 0), 2)
        cv2.putText(annotated, "DOOR", (door_x1 + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(annotated, f"IN: {self.state.entered}  OUT: {self.state.exited}  NOW: {self.state.current_people}",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    def _ai_worker(self):
        while self.state.running:
            try:
                event_id, crop = self._ai_queue.get(timeout=1)
                if crop is None or crop.size == 0:
                    continue

                existing = self.db.get_all_persons_with_descriptions() if self.db else []
                result = self.llm.identify_person(crop, existing)
                pid = result.get("person_id", 0)
                desc = result.get("description", "")

                if pid > 0 and self.db:
                    self.db.assign_event(event_id, pid)
                    self.db.add_face_image(pid, crop)
                    for d in self.tracked.values():
                        if d.get("event_id") == event_id:
                            d["person_id"] = pid
                            break
                else:
                    pid = self.db.resolve_event(event_id, desc)
                    if pid and crop is not None:
                        self.db.add_face_image(pid, crop)
            except queue.Empty:
                continue
            except Exception as e:
                with open("logs/pipeline_error.log", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} _ai_worker: {e}\n")
                    traceback.print_exc(file=f)


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
        counter = PeopleCounter(self.state, self.logger, db=self.state.person_db, llm=llm)

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
