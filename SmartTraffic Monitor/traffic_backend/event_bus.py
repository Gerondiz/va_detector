import sys
import threading
import queue
import traceback
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EventBus")


class EventBus:
    def __init__(self):
        self._subscribers = {}
        self._ai_queue = queue.Queue()
        self._ai_event_types = {"passed_ai"}
        self._ai_worker = threading.Thread(
            target=self._ai_worker_loop,
            name="AI-Worker-Thread",
            daemon=True
        )
        self._ai_worker.start()
        logger.info("EventBus initialized")

    def subscribe(self, event_type: str, callback):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.info(f"Subscriber {callback.__name__} registered for '{event_type}'")

    def publish(self, event_type: str, data: dict):
        if event_type not in self._subscribers or not self._subscribers[event_type]:
            return
        if event_type in self._ai_event_types:
            logger.info(f"AI event '{event_type}' queued (size: {self._ai_queue.qsize() + 1})")
            self._ai_queue.put((event_type, data))
            return
        for callback in self._subscribers[event_type]:
            thread = threading.Thread(
                target=self._safe_execute,
                args=(callback, event_type, data),
                name=f"SubThread-{event_type}",
                daemon=True
            )
            thread.start()

    def _safe_execute(self, callback, event_type: str, data: dict):
        try:
            callback(event_type, data)
        except Exception as e:
            logger.error(f"Error in {callback.__name__} for '{event_type}': {e}")
            traceback.print_exc(file=sys.stderr)

    def _ai_worker_loop(self):
        while True:
            try:
                event_type, data = self._ai_queue.get()
                logger.info(f"AI-Worker processing '{event_type}' track {data.get('track_id')}")
                for callback in self._subscribers.get(event_type, []):
                    self._safe_execute(callback, event_type, data)
                logger.info(f"AI-Worker done '{event_type}'")
                self._ai_queue.task_done()
            except Exception as e:
                logger.critical(f"AI-Worker crash: {e}")
                traceback.print_exc(file=sys.stderr)
