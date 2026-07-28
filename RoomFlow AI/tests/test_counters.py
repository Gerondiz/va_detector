import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
import logging
logging.disable(logging.CRITICAL)

from backend.pipeline import PeopleCounter, SharedState
from backend.event_bus import EventBus


class MockEventBus:
    def __init__(self):
        self.events = []
    def subscribe(self, event_type, callback):
        pass
    def publish(self, event_type, data):
        self.events.append((event_type, data))


def make_person(oid, cx, cy=150, w=30, h=80):
    bbox = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
    return {"label": "person", "id": oid, "bbox": bbox}


def make_frame(h=200, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_state():
    state = SharedState()
    state.door_left, state.door_right = 0.3, 0.7
    state.door_top, state.door_bottom = 0.0, 1.0
    state.entered, state.exited = 0, 0
    return state


def run_steps(counter, frame, steps):
    for s in steps:
        objects = [make_person(**kw) for kw in s.get("persons", [])]
        counter.update(objects, frame, frame.copy())
        if s.get("sleep"):
            time.sleep(s["sleep"])


def test_walk_through_door_publishes_entry():
    """Person walks LEFT→DOOR→RIGHT → entry event published after 0.3s delay."""
    frame = make_frame()
    state = make_state()
    bus = MockEventBus()
    counter = PeopleCounter(state, bus)

    run_steps(counter, frame, [
        {"persons": [{"oid": 1, "cx": 40}]},
        {"persons": [{"oid": 1, "cx": 80}]},
        {"persons": [{"oid": 1, "cx": 150}]},
        {"persons": [{"oid": 1, "cx": 150}], "sleep": 0.4},
        {"persons": [{"oid": 1, "cx": 160}]},
    ])
    published = [d for t, d in bus.events if t == "entered"]
    assert len(published) == 1, f"expected 1 entered event, got {len(published)}"
    assert published[0]["track_id"] == 1
    print("  PASS test_walk_through_door_publishes_entry")


def test_appears_inside_door_publishes_exit():
    """Person appears inside zone and vanishes → exit event published."""
    frame = make_frame()
    state = make_state()
    bus = MockEventBus()
    counter = PeopleCounter(state, bus)

    run_steps(counter, frame, [
        {"persons": [{"oid": 2, "cx": 100}]},
        {"persons": []},
        {"sleep": 0.6},
        {"persons": []},
    ])
    published = [d for t, d in bus.events if t == "exited"]
    assert len(published) == 1, f"expected 1 exited event, got {len(published)}"
    print("  PASS test_appears_inside_door_publishes_exit")


def test_exit_frame_saved():
    """Verify exit_frame/exit_bbox are saved for new tracks INSIDE the door zone."""
    frame = make_frame()
    state = make_state()
    counter = PeopleCounter(state, MockEventBus())
    oid = 3
    objs = [make_person(oid, cx=100)]
    counter.update(objs, frame, frame.copy())

    assert oid in counter.tracked, "track should exist"
    d = counter.tracked[oid]
    assert d["exit_frame"] is not None, f"exit_frame should be saved, got {d.get('exit_frame')}"
    assert d["exit_bbox"] is not None, f"exit_bbox should be saved, got {d.get('exit_bbox')}"
    print("  PASS test_exit_frame_saved")


def test_entry_cooldown():
    """Rapid re-entry within 100px/10s should block second event."""
    frame = make_frame()
    state = make_state()
    bus = MockEventBus()
    counter = PeopleCounter(state, bus)

    run_steps(counter, frame, [
        {"persons": [{"oid": 4, "cx": 40}]},
        {"persons": [{"oid": 4, "cx": 100}]},
        {"persons": [{"oid": 4, "cx": 150}]},
        {"persons": [{"oid": 4, "cx": 150}], "sleep": 0.4},
        {"persons": [{"oid": 4, "cx": 160}]},
        {"persons": [{"oid": 4, "cx": 100}]},
        {"persons": [{"oid": 4, "cx": 150}]},
        {"persons": [{"oid": 4, "cx": 150}], "sleep": 0.4},
        {"persons": [{"oid": 4, "cx": 160}]},
    ])
    entered = [d for t, d in bus.events if t == "entered"]
    # First entry fires, second is suppressed by cooldown (same position ~150px)
    assert len(entered) == 1, f"expected 1 entered (cooldown), got {len(entered)}"
    print("  PASS test_entry_cooldown")


def test_exit_screenshot_uses_saved_frame():
    """Exit screenshot should use exit_frame (not dark current frame)."""
    frame = make_frame()
    frame[:] = (0, 255, 0)
    state = make_state()
    counter = PeopleCounter(state, MockEventBus())

    objs = [make_person(5, cx=100)]
    counter.update(objs, frame, frame.copy())
    d = counter.tracked[5]
    assert d["exit_frame"] is not None
    assert d["exit_frame"][100, 100].tolist() == [0, 255, 0]

    empty_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    counter.update([], empty_frame, empty_frame.copy())
    exit_info = counter._recently_lost[5]
    assert exit_info.get("exit_frame") is not None
    assert exit_info["exit_frame"][100, 100].tolist() == [0, 255, 0]

    print("  PASS test_exit_screenshot_uses_saved_frame")


if __name__ == "__main__":
    print("Running PeopleCounter tests (Event Bus architecture)...")
    test_walk_through_door_publishes_entry()
    test_appears_inside_door_publishes_exit()
    test_exit_frame_saved()
    test_entry_cooldown()
    test_exit_screenshot_uses_saved_frame()
    print("\nAll tests passed!")
