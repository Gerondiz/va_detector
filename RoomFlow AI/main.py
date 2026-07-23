import cv2
import time
import os
from config import CAMERA, LOGS
from detector import Detector
from audio import AudioProcessor
from llm_client import LLMAnalyzer
from logger import EventLogger


class TrackManager:
    def __init__(self, logger: EventLogger | None = None):
        self.active: dict[int, dict] = {}
        self.events: list[str] = []
        self.logger = logger

    def update(self, objects: list[dict]):
        current_ids = {o["id"] for o in objects if o["id"] is not None}
        prev_ids = set(self.active.keys())

        for o in objects:
            if o["id"] is None:
                continue
            if o["id"] not in prev_ids:
                msg = f"[ENTER] {o['label']} (id={o['id']})"
                self.events.append(msg)
                print(f"[{time.strftime('%H:%M:%S')}] {msg}")
                if self.logger:
                    self.logger.object_enter(o["label"], o["id"], o["confidence"])
            self.active[o["id"]] = o

        for oid in prev_ids - current_ids:
            label = self.active[oid]["label"]
            msg = f"[EXIT] {label} (id={oid})"
            self.events.append(msg)
            print(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if self.logger:
                self.logger.object_exit(label, oid)
            del self.active[oid]

        if len(self.events) > 50:
            self.events = self.events[-50:]

    def draw(self, frame):
        y = frame.shape[0] - 30
        for event in self.events[-5:]:
            cv2.putText(frame, event, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y -= 18


def create_capture(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(111, 1)
    return cap


def has_display():
    return os.environ.get("DISPLAY") is not None


def main():
    rtsp_url = CAMERA["rtsp_url"]
    print(f"[INFO] Connecting to {rtsp_url} ...")
    cap = create_capture(rtsp_url)

    if not cap.isOpened():
        print("[ERROR] Cannot open RTSP stream")
        return

    detector = Detector()
    logger = EventLogger()
    tracker = TrackManager(logger)
    audio = AudioProcessor(rtsp_url)
    audio.start()
    llm = LLMAnalyzer()
    llm_last_objects = []
    llm_last_asr = None
    llm_result_display = None
    llm_result_time = 0

    frame_count = 0
    fps_start = time.time()
    show_gui = has_display()

    if show_gui:
        print("[INFO] Display available — press 'q' to exit")
    else:
        print("[INFO] Headless mode — saving frames to logs/")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Lost frame, reconnecting...")
            cap.release()
            time.sleep(1)
            cap = create_capture(rtsp_url)
            continue

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        results = detector.detect(frame, track=True)
        objects = detector.get_objects(results)
        tracker.update(objects)
        annotated = detector.get_annotated_frame(results)

        if time.time() - fps_start >= 1.0:
            fps_display = frame_count
            frame_count = 0
            fps_start = time.time()
            cv2.putText(annotated, f"FPS: {fps_display}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        active_labels = [f"#{o['id']} {o['label']}" for o in tracker.active.values()]
        cv2.putText(annotated, f"Active: {', '.join(active_labels)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        asr_text = audio.get_text()
        if asr_text:
            tracker.events.append(f"[ASR] {asr_text}")
            cv2.putText(annotated, f"Speech: {asr_text}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            logger.asr(asr_text)
            llm_last_asr = asr_text

        if frame_count % 15 == 0 and (objects or llm_last_asr):
            llm.request(objects or llm_last_objects, llm_last_asr)
            llm_last_asr = None

        llm_result = llm.get_result()
        if llm_result:
            llm_result_display = llm_result
            llm_result_time = time.time()
            cls = llm_result.get("classification", "?")
            tracker.events.append(f"[LLM] {cls}: {llm_result.get('reason', '')[:50]}")
            logger.llm_analysis(cls, llm_result.get("reason", ""), llm_result.get("action", ""))

        if llm_result_display and time.time() - llm_result_time < 5:
            r = llm_result_display
            cls = r.get("classification", "?")
            color = {"safe": (0, 255, 0), "threat": (0, 0, 255), "uncertain": (0, 255, 255)}.get(cls, (255, 255, 255))
            reason = r.get("reason", "")[:50]
            cv2.putText(annotated, f"LLM: {cls.upper()}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(annotated, reason, (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        tracker.draw(annotated)

        if LOGS.get("save_screenshots") and objects:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"{LOGS['dir']}/{ts}.jpg"
            cv2.imwrite(path, annotated)
            logger.screenshot(path, objects, asr_text)

        if show_gui:
            cv2.imshow("VA Detector", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            time.sleep(0.03)

    audio.stop()
    cap.release()
    if show_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
