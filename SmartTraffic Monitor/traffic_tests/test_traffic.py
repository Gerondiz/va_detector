import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from traffic_backend.pipeline import TrafficCounter, TrafficState
from traffic_backend.event_bus import EventBus


class MockBus:
    def __init__(self):
        self.events = []
    def subscribe(self, t, cb):
        pass
    def publish(self, t, d):
        self.events.append((t, d))


def make_obj(track_id, cx, cy=100, label="car", cls=2, w=40, h=30):
    return {
        "id": track_id,
        "class": cls,
        "label": label,
        "confidence": 0.9,
        "bbox": [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
    }


def make_frame(w=400, h=300):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_car_passes_right():
    """Car crosses vertical line left→right."""
    state = TrafficState()
    bus = MockBus()
    counter = TrafficCounter(state, bus)
    frame = make_frame()
    annotated = frame.copy()

    steps = [
        [make_obj(1, cx=50)],
        [make_obj(1, cx=150)],
        [make_obj(1, cx=250)],  # crosses line at 200 (line_position=0.5, w=400)
    ]
    for objs in steps:
        counter.update(objs, frame, annotated)
        time.sleep(0.01)

    passed = [d for t, d in bus.events if t == "passed"]
    assert len(passed) == 1, f"expected 1 pass, got {len(passed)}"
    assert passed[0]["direction"] == "right"
    assert passed[0]["vehicle_class"] == "car"
    print("  PASS test_car_passes_right")


def test_car_passes_left():
    """Car crosses vertical line right→left."""
    state = TrafficState()
    bus = MockBus()
    counter = TrafficCounter(state, bus)
    frame = make_frame()
    annotated = frame.copy()

    steps = [
        [make_obj(2, cx=350)],
        [make_obj(2, cx=250)],
        [make_obj(2, cx=50)],   # crosses line at 200
    ]
    for objs in steps:
        counter.update(objs, frame, annotated)
        time.sleep(0.01)

    passed = [d for t, d in bus.events if t == "passed"]
    assert len(passed) == 1, f"expected 1 pass, got {len(passed)}"
    assert passed[0]["direction"] == "left"
    print("  PASS test_car_passes_left")


def test_bus_classification():
    """Bus (COCO class 5) is classified correctly."""
    state = TrafficState()
    bus = MockBus()
    counter = TrafficCounter(state, bus)
    frame = make_frame()
    annotated = frame.copy()

    obj = make_obj(3, cx=50, cls=5, label="bus")
    counter.update([obj], frame, annotated)
    obj["bbox"][0] = 150
    obj["bbox"][2] = 190
    counter.update([obj], frame, annotated)
    obj["bbox"][0] = 250
    obj["bbox"][2] = 290
    counter.update([obj], frame, annotated)

    passed = [d for t, d in bus.events if t == "passed"]
    assert len(passed) == 1
    assert passed[0]["vehicle_class"] == "bus"
    print("  PASS test_bus_classification")


def test_no_double_count():
    """Same track within 10s cooldown at same crossing should not double-count."""
    state = TrafficState()
    bus = MockBus()
    counter = TrafficCounter(state, bus)
    frame = make_frame()
    annotated = frame.copy()

    counter.update([make_obj(4, cx=50)], frame, annotated)
    counter.update([make_obj(4, cx=150)], frame, annotated)
    counter.update([make_obj(4, cx=250)], frame, annotated)

    # cross back again (same track, within 10s)
    counter.update([make_obj(4, cx=350)], frame, annotated)
    counter.update([make_obj(4, cx=150)], frame, annotated)
    counter.update([make_obj(4, cx=50)], frame, annotated)

    passed = [d for t, d in bus.events if t == "passed"]
    assert len(passed) == 2, f"expected 2 passes (forward+back), got {len(passed)}"
    assert passed[0]["direction"] == "right"
    assert passed[1]["direction"] == "left"
    print("  PASS test_no_double_count")


def test_horizontal_line():
    """Horizontal line crossing top→bottom."""
    state = TrafficState()
    state.line_horizontal = True
    bus = MockBus()
    counter = TrafficCounter(state, bus)
    frame = make_frame()
    annotated = frame.copy()

    obj = make_obj(5, cx=200, cy=30)
    counter.update([obj], frame, annotated)
    obj["bbox"][1] = 70
    obj["bbox"][3] = 100
    counter.update([obj], frame, annotated)
    obj["bbox"][1] = 170
    obj["bbox"][3] = 200  # crosses line at 150 (h=300, 0.5)
    counter.update([obj], frame, annotated)

    passed = [d for t, d in bus.events if t == "passed"]
    assert len(passed) == 1
    assert passed[0]["direction"] == "right"  # top→bottom = right
    print("  PASS test_horizontal_line")


if __name__ == "__main__":
    print("Running SmartTraffic tests...")
    test_car_passes_right()
    test_car_passes_left()
    test_bus_classification()
    test_no_double_count()
    test_horizontal_line()
    print("\nAll tests passed!")
