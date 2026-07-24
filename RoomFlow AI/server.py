import cv2
import json
import time
import os
from pathlib import Path
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pipeline import SharedState, CameraPipeline
from database import PersonDB
from config import CAMERA, MODELS

state = SharedState()
app = FastAPI(title="VA Detector")
BASE = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE / "templates"))


def gen_mjpeg():
    while state.running:
        frame = state.get_frame()
        if frame is None:
            time.sleep(0.03)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.03)


@app.on_event("startup")
def startup():
    t = CameraPipeline(state)
    t.start()


@app.on_event("shutdown")
def shutdown():
    state.running = False


@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/logs")
async def logs_page(request: Request):
    return templates.TemplateResponse(request, "logs.html")


@app.get("/people")
async def people_page(request: Request):
    return templates.TemplateResponse(request, "people.html")

@app.get("/people/{person_id}")
async def person_detail(request: Request, person_id: int):
    return templates.TemplateResponse(request, "person_detail.html", {"person_id": person_id})


@app.get("/settings")
async def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html", {
        "camera_ip": CAMERA["ip"],
        "camera_url": CAMERA["rtsp_url"],
        "yolo_model": MODELS["yolo"],
        "llm_model": MODELS["llm"]["model"],
        "llm_url": MODELS["llm"]["base_url"],
        "line_pos": state.line_position,
        "door_left": state.door_left,
        "door_right": state.door_right,
        "door_top": state.door_top,
        "door_bottom": state.door_bottom,
    })


@app.get("/video")
def video():
    return StreamingResponse(gen_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


class LineModel(BaseModel):
    line_position: float

@app.post("/api/set_line")
def api_set_line(data: LineModel):
    state.set_line_position(data.line_position)
    return {"ok": True, "line_position": state.line_position}

class DoorZoneModel(BaseModel):
    door_left: float
    door_right: float
    door_top: float = 0.0
    door_bottom: float = 1.0

@app.post("/api/set_door_zone")
def api_set_door_zone(data: DoorZoneModel):
    state.set_door_zone(data.door_left, data.door_right, data.door_top, data.door_bottom)
    return {"ok": True, "door_left": state.door_left, "door_right": state.door_right,
            "door_top": state.door_top, "door_bottom": state.door_bottom}

@app.get("/api/status")
def api_status():
    return {
        "entered": state.entered,
        "exited": state.exited,
        "current": state.current_people,
        "fps": state.fps,
        "line_position": state.line_position,
        "door_left": state.door_left,
        "door_right": state.door_right,
        "door_top": state.door_top,
        "door_bottom": state.door_bottom,
    }


@app.get("/api/events")
def api_events():
    return state.get_events(50)


@app.get("/api/person_events")
def api_person_events():
    return state.get_person_events(100)


@app.get("/api/people")
def api_people():
    db = state.person_db
    if db is None:
        return {"people": []}
    return {"people": db.get_all_persons()}

@app.get("/api/people/{person_id}")
def api_person_detail(person_id: int):
    db = state.person_db
    if db is None:
        return {"error": "no db"}
    p = db.get_person(person_id)
    if p is None:
        return JSONResponse({"error": "not found"}, 404)
    return p

class RenameModel(BaseModel):
    person_id: int
    name: str

@app.post("/api/people/rename")
def api_people_rename(data: RenameModel):
    db = state.person_db
    if db is not None:
        db.rename(data.person_id, data.name)
    return {"ok": True}

@app.get("/api/screenshots")
def api_screenshots():
    log_dir = BASE / "logs"
    files = sorted(log_dir.glob("*.jpg"), key=os.path.getmtime, reverse=True)
    return [f.name for f in files[:100]]


@app.get("/screenshots/{filename}")
def serve_screenshot(filename: str):
    path = BASE / "logs" / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, 404)
    return FileResponse(path, media_type="image/jpeg")

@app.get("/face_images/{path:path}")
def serve_face_image(path: str):
    full = BASE / "logs" / "faces" / path
    if not full.exists():
        return JSONResponse({"error": "not found"}, 404)
    return FileResponse(full, media_type="image/jpeg")



