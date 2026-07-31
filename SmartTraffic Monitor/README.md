# SmartTraffic Monitor

*Проект в составе репозитория `va_detector`. Файлы проекта — в папке `SmartTraffic Monitor/`.*

Real-time counting of vehicles (car, bus, truck, motorcycle) with direction detection, using a camera pointed at the road.

## How it works

1. **Capture** — OpenCV reads MJPEG/RTSP stream frame by frame
2. **Detection** — YOLOv8n detects vehicles in each frame (COCO classes: car, bus, truck, motorcycle)
3. **Tracking** — ByteTrack assigns stable IDs across frames
4. **Counting line** — configurable line (vertical/horizontal, position 0–1) on the frame; a pass is counted when a track's center crosses the line
5. **ID stabilization** — IoU matching merges re-assigned track IDs, so the same vehicle is not double-counted
6. **Event flow** — TrafficCounter publishes `passed` events via the Event Bus; LoggerSubscriber increments counters and writes to SQLite
7. **Web UI** — live MJPEG stream with annotated boxes + counters per class/direction

## Requirements

- Python 3.10+
- Camera with MJPEG or RTSP stream (any IP webcam / WebcamXP / insecam)
- `yolov8n.pt` model (downloaded automatically on first run)

## Setup

```bash
pip install -r requirements.txt
```

The camera URL can be changed at runtime via the **Settings** page (no restart needed — the pipeline reconnects automatically).

For a WebcamXP camera the stream is usually at `http://<ip>:8080/cam_1.cgi`.

## Run

```bash
./run_server.sh
```

Opens at `http://localhost:8001`. The watchdog loop automatically restarts the server if it crashes.

## Pages

| Route | Description |
|-------|-------------|
| `/` | Live video + counters by class/direction + recent passes feed |
| `/settings` | Counting line (position + orientation), detection confidence, camera URL, counters reset |

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | All counters, FPS, line settings, confidence |
| GET | `/api/events` | Recent 50 passes (from DB) |
| GET | `/api/stats` | Daily statistics (default last 7 days) |
| POST | `/api/set_line` | Set counting line `{line_position, line_horizontal}` |
| POST | `/api/set_setting` | Save `{key, value}` (line_position, confidence, camera_url, …) |
| POST | `/api/reset` | Reset all counters and events |
| GET | `/video` | MJPEG live stream |

## Architecture

```
Camera (MJPEG/RTSP)
  → OpenCV → frame
  → YOLOv8n + ByteTrack → detection + tracking
  → TrafficCounter → line crossing + direction + IoU ID merge → passed events
  → EventBus (Pub/Sub)
      └── LoggerSubscriber → SQLite (events + daily_stats) + in-memory counters
  → FastAPI + Jinja2 → web UI (dashboard + settings)
```

## Project structure

```
SmartTraffic Monitor/
├── traffic_backend/
│   ├── main.py          FastAPI server + routes + MJPEG stream
│   ├── pipeline.py      TrafficPipeline, TrafficCounter, TrafficState, subscribers
│   ├── detector.py      YOLOv8 + ByteTrack wrapper
│   ├── event_bus.py     Pub/Sub event bus (reused from RoomFlow AI)
│   ├── database.py      TrafficDB (events, daily_stats, settings)
│   └── config.py        Camera and model configuration
├── traffic_templates/
│   ├── base.html
│   ├── dashboard.html
│   └── settings.html
├── traffic_tests/
│   └── test_traffic.py
├── Docs/
│   └── ТЗ.md            Technical specification (Russian)
├── run_server.sh        Watchdog launch script (port 8001)
└── requirements.txt
```

## Testing

```bash
cd "SmartTraffic Monitor" && python3 traffic_tests/test_traffic.py
```
