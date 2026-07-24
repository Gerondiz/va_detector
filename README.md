# RoomFlow AI

*Проект в составе репозитория `va_detector`. Файлы проекта — в папке `RoomFlow AI/`.*

Real-time people counting and AI‑based person re‑identification using a ceiling‑mounted camera pointed at a doorway.

## How it works

1. **Detection** — YOLOv8 detects people in each frame
2. **Tracking** — ByteTrack assigns stable IDs across frames
3. **Door zone** — configurable rectangle over the doorway; entry/exit is determined by a person's feet entering or leaving this zone
4. **Identification** — each detected person is cropped and sent to a vision LLM (moondream) for description; descriptions are matched against known persons via Jaccard similarity
5. **Face matching** — insightface extracts face embeddings for fast re‑identification on re‑entry
6. **Persistence** — SQLite stores persons, events, face images, and settings

## Requirements

- Python 3.10+
- Camera (tested with D‑Link DCS‑2102 MJPEG)
- Ollama with `moondream:latest` (optional, for AI descriptions)

## Setup

```bash
pip install -r requirements.txt
```

Edit `backend/config.py` if your camera URL differs.

## Run

```bash
./run_server.sh
```

Opens at `http://localhost:8000`.

The watchdog loop automatically restarts the server if the pipeline crashes (known OpenCV MJPEG issue).

## Pages

| Route | Description |
|-------|-------------|
| `/` | Live video + counters + event log |
| `/people` | All known persons as cards |
| `/people/{id}` | Person detail: AI description, face gallery, event timeline with screenshots |
| `/settings` | Door zone sliders, model info, counter reset |
| `/logs` | Pipeline error log |

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Current counts, FPS, door zone |
| GET | `/api/events` | Recent text events |
| GET | `/api/people` | List of known persons |
| GET | `/api/people/{id}` | Person details + events |
| POST | `/api/people/rename` | Rename a person |
| POST | `/api/set_door_zone` | Update door zone rectangle |
| POST | `/api/reset_counters` | Reset entered/exited to 0 |
| GET | `/video` | MJPEG live stream |
| GET | `/screenshots/{file}` | Event screenshot |
| GET | `/face_images/{path}` | Face image |

## Project structure

```
RoomFlow AI/
├── backend/
│   ├── main.py          FastAPI server + routes
│   ├── pipeline.py      Camera pipeline, tracking, counting
│   ├── database.py      SQLite models + settings
│   ├── detector.py      YOLOv8 + ByteTrack wrapper
│   ├── llm_client.py    Moondream vision API client
│   ├── config.py        Camera and model configuration
│   ├── logger.py        Event logger (legacy)
│   └── audio.py         Microphone capture (disabled)
├── templates/           Jinja2 pages
├── static/              Static assets
├── logs/                Screenshots and error logs
├── run_server.sh        Watchdog launch script
└── requirements.txt
```
