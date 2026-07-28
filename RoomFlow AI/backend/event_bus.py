import threading
import time
import traceback


class EventBus:
    def __init__(self):
        self._subs: dict[str, list] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, callback):
        with self._lock:
            self._subs.setdefault(event_type, []).append(callback)

    def publish(self, event_type: str, data: dict):
        with self._lock:
            cbs = list(self._subs.get(event_type, []))
        for cb in cbs:
            threading.Thread(target=self._safe_call, args=(cb, event_type, data), daemon=True).start()

    def _safe_call(self, cb, event_type, data):
        try:
            cb(event_type, data)
        except Exception as e:
            with open("logs/pipeline_error.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} EventBus: {e}\n")
                traceback.print_exc(file=f)
