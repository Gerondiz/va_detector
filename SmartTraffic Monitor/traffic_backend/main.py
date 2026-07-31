import cv2
import time
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .pipeline import TrafficState, TrafficPipeline
from .database import TrafficDB, get_setting, set_setting
from .config import CAMERA, DEFAULT_CONFIDENCE

state = TrafficState()
app = FastAPI(title="SmartTraffic Monitor")

BASE = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE / "traffic_templates"))

if (BASE / "traffic_static").exists():
    app.mount("/static", StaticFiles(directory=str(BASE / "traffic_static")), name="static")


def gen_mjpeg():
    while state.running:
        frame = state.get_frame()
        if frame is None:
            time.sleep(0.03)
            continue
        if time.time() - state.last_frame_time > 3.0:
            break
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.03)


@app.on_event("startup")
def startup():
    state.db = TrafficDB()
    state.camera_url = get_setting("camera_url", CAMERA["mjpg_url"])
    try:
        state.line_position = float(get_setting("line_position", "0.5"))
        state.line_horizontal = get_setting("line_horizontal", "false") == "true"
        state.confidence = float(get_setting("confidence", str(DEFAULT_CONFIDENCE)))
    except Exception:
        pass
    t = TrafficPipeline(state)
    t.start()


@app.on_event("shutdown")
def shutdown():
    state.running = False


@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/settings")
async def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html", {
        "line_position": state.line_position,
        "line_horizontal": state.line_horizontal,
        "confidence": state.confidence,
        "camera_url": state.camera_url,
    })


@app.get("/video")
def video():
    return StreamingResponse(gen_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


class LineModel(BaseModel):
    line_position: float
    line_horizontal: bool = False


@app.post("/api/set_line")
def api_set_line(data: LineModel):
    state.line_position = max(0.0, min(1.0, data.line_position))
    state.line_horizontal = data.line_horizontal
    set_setting("line_position", str(state.line_position))
    set_setting("line_horizontal", str(state.line_horizontal))
    return {"ok": True}


class SettingsModel(BaseModel):
    key: str
    value: str


@app.post("/api/set_setting")
def api_set_setting(data: SettingsModel):
    set_setting(data.key, data.value)
    if data.key == "line_position":
        state.line_position = float(data.value)
    elif data.key == "line_horizontal":
        state.line_horizontal = data.value == "true"
    elif data.key == "confidence":
        state.confidence = float(data.value)
    elif data.key == "camera_url":
        state.camera_url = data.value
        state.reconnect = True
    return {"ok": True}


@app.get("/api/status")
def api_status():
    return {
        "car_left": state.car_left,
        "car_right": state.car_right,
        "bus_left": state.bus_left,
        "bus_right": state.bus_right,
        "truck_left": state.truck_left,
        "truck_right": state.truck_right,
        "moto_left": state.moto_left,
        "moto_right": state.moto_right,
        "fps": state.fps,
        "line_position": state.line_position,
        "line_horizontal": state.line_horizontal,
        "confidence": state.confidence,
    }


@app.get("/api/events")
def api_events():
    if state.db is None:
        return []
    return state.db.get_recent_events(50)


@app.get("/api/stats")
def api_stats(days: int = 7):
    if state.db is None:
        return []
    return state.db.get_stats(days)


@app.post("/api/reset")
def api_reset():
    if state.db is not None:
        state.db.clear_events()
    for attr in ("car_left", "car_right", "bus_left", "bus_right",
                 "truck_left", "truck_right", "moto_left", "moto_right"):
        setattr(state, attr, 0)
    return {"ok": True}



