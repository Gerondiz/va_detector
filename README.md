# RoomFlow AI

*Проект в составе репозитория `va_detector`. Файлы проекта — в папке `RoomFlow AI/`.*

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

## Architecture

```
Camera (MJPEG)
  → YOLOv8n + ByteTrack → detection + tracking
  → PeopleCounter → door zone logic → entry/exit events
  → EventBus (Pub/Sub)
      ├── LoggerSubscriber → SQLite + AI event
      └── AI-Worker-Thread (queue) → LM Studio → PersonDB
```

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

## Testing

```bash
cd "RoomFlow AI" && python3 tests/test_counters.py
cd "RoomFlow AI" && python3 tests/test_vision.py <image_path>
```
