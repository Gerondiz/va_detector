# va_detector

Monorepo with two real-time computer-vision projects:

| Project | Purpose | Port |
|---------|---------|------|
| **RoomFlow AI** | People counting + AI person re-identification (ceiling camera at a doorway) | 8000 |
| **SmartTraffic Monitor** | Vehicle counting by class and direction (road camera) | 8001 |

---

# RoomFlow AI

*Файлы проекта — в папке `RoomFlow AI/`.*

Real-time people counting and AI‑based person re‑identification using a ceiling‑mounted camera pointed at a doorway.

## How it works

1. **Detection** — YOLOv8n detects people in each frame
2. **Tracking** — ByteTrack assigns stable IDs across frames
3. **Door zone** — configurable rectangle over the doorway; entry/exit determined by person's lower bbox edge (y2) entering/leaving this zone
4. **Entry/exit confirmation** — 0.3s pending delay for entry, 0.5s for exit (prevents flicker); positional cooldown (100px / 10s)
5. **Event Bus (Pub/Sub)** — PeopleCounter publishes `entered`/`exited` events; LoggerSubscriber writes to SQLite + publishes AI events; AI-Worker-Thread processes vision sequentially
6. **AI identification** — crop sent to LM Studio (gemma4:e4b) in a single request: system context (known persons) + image + task prompt → JSON `{"person_id", "description"}`
7. **Persistence** — SQLite stores events, persons, face images, settings

## Requirements

- Python 3.10+
- Camera with MJPEG stream (tested with D‑Link DCS‑2102)
- LM Studio or Ollama with vision-capable model (gemma4, moondream, etc.)

## Setup

```bash
pip install -r requirements.txt
```

Edit `backend/config.py` if your camera URL or LLM server differs. Settings can also be changed at runtime via the web UI (Settings page).

## Run

```bash
./run_server.sh
```

Opens at `http://localhost:8000`. The watchdog loop automatically restarts the server if the pipeline crashes (known OpenCV MJPEG issue).

## Pages

| Route | Description |
|-------|-------------|
| `/` | Live video + counters + event feed + person crossings table |
| `/people` | All known persons with face thumbs + screenshots |
| `/people/{id}` | Person detail: AI description, face gallery, event timeline |
| `/settings` | Door zone sliders, LLM URL + Check, vision model, counters reset, person management |
| `/logs` | Event screenshots gallery |

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Current counts, FPS, door zone |
| GET | `/api/events` | Recent 50 text events (from DB) |
| GET | `/api/person_events` | Recent 100 events for dashboard |
| GET | `/api/people` | List of known persons (with `?date=` filter) |
| GET | `/api/people/{id}` | Person details + events + face images |
| POST | `/api/people/rename` | Rename a person |
| POST | `/api/set_setting` | Save `{key, value}` to settings |
| POST | `/api/set_door_zone` | Update door zone rectangle |
| POST | `/api/check_llm` | Check LLM server availability |
| POST | `/api/reset_counters` | Reset entered/exited to 0 |
| POST | `/api/reset_persons` | Delete all persons and reassign events |
| POST | `/api/dedup_persons` | LLM-based person deduplication |
| GET | `/video` | MJPEG live stream |
| GET | `/api/screenshots` | List of recent screenshots |
| GET | `/screenshots/{path}` | Event screenshot |
| GET | `/face_images/{path}` | Face image |

## Project structure

```
RoomFlow AI/
├── backend/
│   ├── main.py          FastAPI server + routes
│   ├── pipeline.py      Camera pipeline, tracking, counting, subscribers
│   ├── event_bus.py     Pub/Sub event bus with AI queue
│   ├── database.py      SQLite models + settings
│   ├── detector.py      YOLOv8 + ByteTrack wrapper
│   ├── llm_client.py    LM Studio vision + JSON client
│   └── config.py        Camera and model configuration
├── templates/           Jinja2 pages (6 files)
├── tests/
│   ├── test_counters.py Entry/exit logic tests
│   └── test_vision.py   Vision model check
├── Docs/
│   ├── ТЗ.md            Technical specification (Russian)
│   ├── event_bus.md     Event Bus architecture docs
│   └── report.md        Full project analysis
├── logs/                Screenshots, crops, pipeline_errors
├── run_server.sh        Watchdog launch script
└── requirements.txt
```

---

# SmartTraffic Monitor

*Файлы проекта — в папке `SmartTraffic Monitor/`.*

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
