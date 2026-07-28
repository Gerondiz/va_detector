import cv2, time, os, threading, traceback
import numpy as np
from dataclasses import dataclass, field
from .config import CAMERA, LOGS
from .detector import Detector
from .llm_client import LLMAnalyzer
from .database import PersonDB, get_setting, set_setting
from .event_bus import EventBus


@dataclass
class SharedState:
    frame: np.ndarray | None = None
    entered: int = 0
    exited: int = 0
    current_people: int = 0
    running: bool = True
    door_left: float = 0.3
    door_right: float = 0.7
    door_top: float = 0.0
    door_bottom: float = 1.0
    fps: int = 0
    person_db: PersonDB | None = None
    last_frame_time: float = 0.0
    llm_status: dict = field(default_factory=lambda: {"ok": False, "error": "not checked"})
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.door_left = float(get_setting("door_left", "0.3"))
        self.door_right = float(get_setting("door_right", "0.7"))
        self.door_top = float(get_setting("door_top", "0.0"))
        self.door_bottom = float(get_setting("door_bottom", "1.0"))

    def update_frame(self, f: np.ndarray):
        with self._lock:
            self.frame = f.copy()
            self.last_frame_time = time.time()

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

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


def _save_date_img(subdir: str, prefix: str, frame: np.ndarray) -> str:
    date_str = time.strftime("%Y-%m-%d")
    dir_path = os.path.join(subdir, date_str)
    os.makedirs(dir_path, exist_ok=True)
    ts = time.strftime("%H%M%S")
    path = os.path.join(dir_path, f"{prefix}_{ts}.jpg")
    ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return f"{date_str}/{prefix}_{ts}.jpg" if ok else ""


def _save_crop_img(prefix: str, frame: np.ndarray) -> str:
    date_str = time.strftime("%Y-%m-%d")
    dir_path = os.path.join(LOGS["dir"], date_str, "crops")
    os.makedirs(dir_path, exist_ok=True)
    ts = time.strftime("%H%M%S")
    fname = f"{prefix}_{ts}.jpg"
    path = os.path.join(dir_path, fname)
    ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return f"{date_str}/crops/{fname}" if ok else ""


class PeopleCounter:
    def __init__(self, state: SharedState, event_bus: EventBus):
        self.state = state
        self.event_bus = event_bus
        self.tracked: dict[int, dict] = {}
        self._pos_cooldown: list[tuple[int, int, float]] = []
        self._recently_lost: dict[int, dict] = {}

    def _match_lost(self, x1, y1, x2, y2) -> int | None:
        now = time.time()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        best = None
        best_px = 200
        for lid, info in self._recently_lost.items():
            if now - info.get("lost_time", 0) < 2.0:
                continue
            lx1, ly1, lx2, ly2 = info.get("lost_bbox", (0, 0, 0, 0))
            dist = ((cx - (lx1 + lx2) / 2) ** 2 + (cy - (ly1 + ly2) / 2) ** 2) ** 0.5
            if dist < best_px:
                best_px = dist
                best = lid
        return best

    def update(self, objects: list[dict], frame: np.ndarray, annotated: np.ndarray):
        h, w = frame.shape[:2]
        dx1 = int(w * self.state.door_left)
        dx2 = int(w * self.state.door_right)
        dy1 = int(h * self.state.door_top)
        dy2 = int(h * self.state.door_bottom)

        persons = [o for o in objects if o["label"] == "person" and o["id"] is not None]
        current_ids = {o["id"] for o in persons}
        now = time.time()

        for o in persons:
            oid = o["id"]
            x1, y1, x2, y2 = o["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            in_door = dx1 <= cx <= dx2 and dy1 <= y2 <= dy2

            if oid not in self.tracked:
                matched = self._match_lost(x1, y1, x2, y2)
                if matched is not None:
                    src = self._recently_lost.pop(matched)
                    self.tracked[oid] = {
                        "prev_in_door": src.get("in_door", in_door),
                        "in_door": in_door,
                        "person_id": src.get("person_id"),
                        "exit_frame": src.get("exit_frame"),
                        "exit_bbox": src.get("exit_bbox"),
                    }
                else:
                    self.tracked[oid] = {"prev_in_door": in_door, "in_door": in_door}
                    if in_door:
                        self.tracked[oid]["exit_frame"] = annotated.copy()
                        self.tracked[oid]["exit_bbox"] = (int(x1), int(y1), int(x2), int(y2))
                d = self.tracked[oid]
            else:
                d = self.tracked[oid]
                was = d["in_door"]

                if was and not in_door:
                    if not d.get("pending_entry"):
                        d["pending_entry"] = True
                        d["pending_entry_time"] = now
                        d["entry_frame"] = annotated.copy()
                        d["entry_bbox"] = (int(x1), int(y1), int(x2), int(y2))

                if not was and in_door:
                    d["exit_frame"] = annotated.copy()
                    d["exit_bbox"] = (int(x1), int(y1), int(x2), int(y2))

                d["prev_in_door"] = was
                d["in_door"] = in_door

            d["_last_bbox"] = (x1, y1, x2, y2)

            label = f"P{d.get('person_id') or ''}" if d.get("person_id") else f"#{oid}"
            cv2.circle(annotated, (int(cx), int(y2)), 4, (0, 255, 255), -1)
            cv2.putText(annotated, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # --- confirm pending entries (0.3s delay, like exit) ---
        for oid, d in list(self.tracked.items()):
            pt = d.get("pending_entry_time")
            if d.get("pending_entry") and pt is not None:
                if d["in_door"]:
                    d["pending_entry"] = False
                elif now - pt > 0.3:
                    d["pending_entry"] = False
                    cx = (d["_last_bbox"][0] + d["_last_bbox"][2]) / 2
                    cy = (d["_last_bbox"][1] + d["_last_bbox"][3]) / 2
                    self._pos_cooldown = [(px, py, pt) for (px, py, pt) in self._pos_cooldown if now - pt < 10]
                    too_soon = any(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 < 100 for (px, py, pt) in self._pos_cooldown)
                    if not too_soon:
                        self._pos_cooldown.append((int(cx), int(cy), now))
                        entry_img = d.get("entry_frame")
                        screenshot = _save_date_img(LOGS["dir"], f"person_{oid}_entered", entry_img) if entry_img is not None else ""
                        entry_bbox = d.get("entry_bbox")
                        crop_path = ""
                        if entry_bbox is not None:
                            x1i, y1i, x2i, y2i = entry_bbox
                            crop = frame[max(0, y1i):min(h, y2i), max(0, x1i):min(w, x2i)]
                            if crop.size > 0:
                                crop_path = _save_crop_img(f"crop_oid{oid}", crop)
                        self.event_bus.publish("entered", {
                            "track_id": oid,
                            "screenshot": screenshot,
                            "crop": crop_path,
                            "bbox": entry_bbox,
                        })

        for oid in list(self.tracked.keys()):
            if oid in current_ids:
                continue
            d = self.tracked.pop(oid)
            self._recently_lost[oid] = {
                **d,
                "lost_time": now,
                "lost_bbox": d.get("_last_bbox", (0, 0, 0, 0)),
                "pending_exit": d.get("in_door", False) or d.get("prev_in_door", False),
            }

        for lid in list(self._recently_lost.keys()):
            info = self._recently_lost[lid]

            if info.get("pending_exit") and now - info.get("lost_time", 0) > 0.5:
                info["pending_exit"] = False
                exit_img = info.get("exit_frame") if info.get("exit_frame") is not None else annotated
                screenshot = _save_date_img(LOGS["dir"], f"person_{lid}_exited", exit_img)
                pid = info.get("person_id")
                exit_bbox = info.get("exit_bbox")
                crop_path = ""
                if pid is None and exit_bbox is not None:
                    x1i, y1i, x2i, y2i = exit_bbox
                    he, we = exit_img.shape[:2]
                    crop = exit_img[max(0, y1i):min(he, y2i), max(0, x1i):min(we, x2i)]
                    if crop.size > 0:
                        crop_path = _save_crop_img(f"exit_oid{lid}", crop)
                self.event_bus.publish("exited", {
                    "track_id": lid,
                    "screenshot": screenshot,
                    "crop": crop_path,
                    "person_id": pid,
                    "bbox": exit_bbox,
                })

            if now - info.get("lost_time", 0) > 5.0:
                del self._recently_lost[lid]

        self.state.current_people = 0
        cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), (255, 255, 0), 2)
        cv2.putText(annotated, "DOOR", (dx1 + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(annotated, f"IN: {self.state.entered}  OUT: {self.state.exited}  NOW: {self.state.current_people}",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


# ---------- Event Bus Subscribers ----------

def make_logger_subscriber(state: SharedState, db: PersonDB, event_bus: EventBus):
    def _handle(event_type: str, data: dict):
        if event_type == "entered":
            state.entered += 1
            set_setting("entered", str(state.entered))
            eid = db.save_entry_event(
                data["track_id"], data.get("crop", ""),
                data["screenshot"], state.entered, state.exited,
            )
            event_bus.publish("entered_ai", {**data, "event_id": eid})
        elif event_type == "exited":
            state.exited += 1
            set_setting("exited", str(state.exited))
            eid = db.save_exit_event(
                data["track_id"], data.get("person_id"),
                data["screenshot"], data.get("crop", ""),
                state.entered, state.exited,
            )
            if data.get("crop"):
                event_bus.publish("exited_ai", {**data, "event_id": eid})
    return _handle


def make_ai_subscriber(db: PersonDB, llm: LLMAnalyzer):
    def _handle(event_type: str, data: dict):
        event_id = data.get("event_id")
        if event_id is None:
            return
        crop_path = data.get("crop")
        if not crop_path:
            return
        ev = db.get_event(event_id)
        if ev is not None and ev.get("person_id") is not None:
            return
        img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", crop_path)
        crop = cv2.imread(img_path)
        if crop is None or crop.size == 0:
            return
        existing = db.get_all_persons_with_descriptions()
        result = llm.identify_person(crop, existing)
        pid = result.get("person_id", 0)
        desc = result.get("description", "")
        if pid > 0:
            db.assign_event(event_id, pid)
            db.add_face_image(pid, crop)
        else:
            pid = db.resolve_event(event_id, desc)
            if pid:
                db.add_face_image(pid, crop)
    return _handle


# ---------- Pipeline ----------

class CameraPipeline(threading.Thread):
    def __init__(self, state: SharedState):
        super().__init__(daemon=True)
        self.state = state

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
        db = self.state.person_db
        state = self.state

        state.entered = db.count_events("entered")
        state.exited = db.count_events("exited")

        event_bus = EventBus()
        llm = LLMAnalyzer()
        counter = PeopleCounter(state, event_bus)
        event_bus.subscribe("entered", make_logger_subscriber(state, db, event_bus))
        event_bus.subscribe("exited", make_logger_subscriber(state, db, event_bus))
        event_bus.subscribe("entered_ai", make_ai_subscriber(db, llm))
        event_bus.subscribe("exited_ai", make_ai_subscriber(db, llm))

        frame_count = 0
        fps_start = time.time()

        while state.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    time.sleep(1)
                    cap = self._connect()
                    if cap is None:
                        time.sleep(2)
                    continue

                frame_count += 1

                results = detector.detect(frame, track=True)
                objects = detector.get_objects(results)
                annotated = detector.get_annotated_frame(results)
                counter.update(objects, frame, annotated)

                cv2.putText(annotated, f"FPS: {state.fps}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                state.update_frame(annotated)

                if time.time() - fps_start >= 1.0:
                    state.fps = frame_count
                    frame_count = 0
                    fps_start = time.time()

            except Exception as e:
                with open("logs/pipeline_error.log", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {e}\n")
                    traceback.print_exc(file=f)
                print(f"[ERROR] Pipeline: {e}")
                time.sleep(2)

        cap.release()
