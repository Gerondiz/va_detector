import json
import time
import os
from config import LOGS


class EventLogger:
    def __init__(self):
        self.log_dir = LOGS.get("dir", "logs")
        self.filepath = os.path.join(self.log_dir, "events.jsonl")
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, event_type: str, data: dict):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "type": event_type,
            "data": data,
        }
        with open(self.filepath, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def object_enter(self, label: str, obj_id: int, confidence: float):
        self.log("object_enter", {"label": label, "id": obj_id, "confidence": confidence})

    def object_exit(self, label: str, obj_id: int):
        self.log("object_exit", {"label": label, "id": obj_id})

    def asr(self, text: str):
        self.log("asr", {"text": text})

    def llm_analysis(self, classification: str, reason: str, action: str):
        self.log("llm_analysis", {
            "classification": classification,
            "reason": reason,
            "action": action,
        })

    def screenshot(self, path: str, objects: list[dict], asr_text: str | None = None):
        self.log("screenshot", {
            "path": path,
            "objects": len(objects),
            "asr": asr_text,
        })
