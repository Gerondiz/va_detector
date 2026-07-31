import cv2
import time
import threading
import traceback
import numpy as np
from dataclasses import dataclass, field
from .config import CAMERA, DEFAULT_CONFIDENCE, DEFAULT_CLASSES
from .detector import Detector
from .event_bus import EventBus
from .database import TrafficDB, get_setting, set_setting


COCO_LABELS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class TrafficState:
    frame: np.ndarray | None = None
    running: bool = True
    line_position: float = 0.5
    line_horizontal: bool = False
    confidence: float = DEFAULT_CONFIDENCE
    active_classes: list[int] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    camera_url: str = ""
    fps: int = 0
    car_left: int = 0
    car_right: int = 0
    bus_left: int = 0
    bus_right: int = 0
    truck_left: int = 0
    truck_right: int = 0
    moto_left: int = 0
    moto_right: int = 0
    db: TrafficDB | None = None
    last_frame_time: float = 0.0
    reconnect: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_frame(self, f: np.ndarray):
        with self._lock:
            self.frame = f.copy()
            self.last_frame_time = time.time()

    def get_frame(self):
        with self._lock:
            if self.frame is not None:
                return self.frame.copy()
        h, w = 480, 640
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No camera feed", (w//2-120, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        cv2.putText(placeholder, "Set Camera URL in Settings", (w//2-160, h//2+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
        return placeholder

    def _class_key(self, cls_name: str, direction: str) -> str:
        return f"{cls_name}_{direction}"

    def inc(self, cls_name: str, direction: str):
        key = self._class_key(cls_name, direction)
        if hasattr(self, key):
            setattr(self, key, getattr(self, key) + 1)


# ── Utility: IoU ──────────────────────────────────────────

def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (a[2] - a[0]) * (a[3] - a[1])
    area2 = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area1 + area2 - inter)


class TrafficCounter:
    def __init__(self, state: TrafficState, event_bus: EventBus):
        self.state = state
        self.event_bus = event_bus
        self._tracks: dict[int, dict] = {}
        self._last_dir: dict[int, str] = {}
        self._lost: list[dict] = []

    def _get_line_px(self, w: int, h: int):
        if self.state.line_horizontal:
            return int(h * self.state.line_position)
        return int(w * self.state.line_position)

    def update(self, objects: list[dict], frame: np.ndarray, annotated: np.ndarray):
        h, w = frame.shape[:2]
        line_px = self._get_line_px(w, h)
        now = time.time()

        current_ids = set()
        for o in objects:
            oid = o["id"]
            if oid is None:
                continue
            current_ids.add(oid)
            x1, y1, x2, y2 = o["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            label = o["label"]
            cls_id = o["class"]
            conf = o["confidence"]

            if self.state.line_horizontal:
                crossed = cy
            else:
                crossed = cx

            if oid in self._tracks:
                prev = self._tracks[oid]
                prev_crossed = prev["crossed"]
                prev_side = "right" if prev_crossed >= line_px else "left"
                cur_side = "right" if crossed >= line_px else "left"

                if prev_side != cur_side:
                    prev_dir = self._last_dir.get(oid)
                    if prev_dir != cur_side:
                        self._last_dir[oid] = cur_side
                        cls_name = COCO_LABELS.get(cls_id, label)
                        self.event_bus.publish("passed", {
                            "vehicle_class": cls_name,
                            "direction": cur_side,
                            "track_id": oid,
                            "confidence": conf,
                        })

                self._tracks[oid] = {"crossed": crossed, "label": label, "bbox": o["bbox"]}
            else:
                # IOU matching with recently lost tracks
                matched = None
                for lost in self._lost:
                    if _iou(o["bbox"], lost["bbox"]) > 0.4:
                        matched = lost
                        break
                if matched:
                    self._lost.remove(matched)
                    self._tracks[oid] = {"crossed": crossed, "label": label, "bbox": o["bbox"]}
                    if "last_dir" in matched:
                        self._last_dir[oid] = matched["last_dir"]
                else:
                    self._tracks[oid] = {"crossed": crossed, "label": label, "bbox": o["bbox"]}

            cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 255, 255), -1)
            cv2.putText(annotated, str(oid), (int(cx)-10, int(cy)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # cleanup stale tracks — keep last bbox for IOU matching
        gone = set(self._tracks.keys()) - current_ids
        for oid in gone:
            t = self._tracks.pop(oid)
            last_dir = self._last_dir.pop(oid, None)
            if "bbox" in t:
                self._lost.append({"bbox": t["bbox"], "last_dir": last_dir})
        self._lost = self._lost[-50:]

        # draw counting line
        color = (0, 255, 0)
        if self.state.line_horizontal:
            cv2.line(annotated, (0, line_px), (w, line_px), color, 2)
        else:
            cv2.line(annotated, (line_px, 0), (line_px, h), color, 2)


# ---------- Subscribers ----------

def make_logger_subscriber(state: TrafficState, db: TrafficDB):
    def _handle(event_type: str, data: dict):
        if event_type == "passed":
            cls_name = data["vehicle_class"]
            direction = data["direction"]
            state.inc(cls_name, direction)
            db.save_pass_event(
                cls_name, direction,
                data.get("confidence", 0.0),
                data.get("track_id", 0),
            )
    return _handle


# ---------- Pipeline ----------

class TrafficPipeline(threading.Thread):
    def __init__(self, state: TrafficState):
        super().__init__(daemon=True)
        self.state = state

    def _connect(self) -> cv2.VideoCapture | None:
        url = self.state.camera_url or CAMERA["mjpg_url"]
        if not url:
            return None
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        if cap.isOpened():
            print(f"[OK] Camera connected: {url}")
            return cap
        cap.release()
        return None

    def run(self):
        cap = None
        while self.state.running and cap is None:
            cap = self._connect()
            if cap is None:
                print(f"[WARN] Camera unavailable ({self.state.camera_url or CAMERA['mjpg_url']}), retry in 3s...")
                time.sleep(3)

        detector = Detector()
        db = TrafficDB()
        self.state.db = db
        event_bus = EventBus()
        counter = TrafficCounter(self.state, event_bus)
        event_bus.subscribe("passed", make_logger_subscriber(self.state, db))

        # load counts from DB
        totals = db.get_totals()
        for key, val in totals.items():
            parts = key.split("_")
            if len(parts) == 2 and hasattr(self.state, key):
                setattr(self.state, key, val)

        frame_count = 0
        fps_start = time.time()

        while self.state.running:
            if self.state.reconnect:
                self.state.reconnect = False
                cap.release()
                time.sleep(0.5)
                cap = self._connect()
                if cap is None:
                    time.sleep(2)
                    continue
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
                results = detector.detect(
                    frame,
                    conf=self.state.confidence,
                    track=True,
                    classes=self.state.active_classes,
                )
                objects = detector.get_objects(results)
                annotated = detector.get_annotated_frame(results)
                counter.update(objects, frame, annotated)

                cv2.putText(annotated, f"FPS: {self.state.fps}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                self.state.update_frame(annotated)

                if time.time() - fps_start >= 1.0:
                    self.state.fps = frame_count
                    frame_count = 0
                    fps_start = time.time()

            except Exception as e:
                with open("/tmp/smarttraffic_error.log", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {e}\n")
                    traceback.print_exc(file=f)
                print(f"[ERROR] Pipeline: {e}")
                time.sleep(2)

        cap.release()
