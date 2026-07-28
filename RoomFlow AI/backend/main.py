import cv2
import json
import time
import os
import requests
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .pipeline import SharedState, CameraPipeline
from .database import PersonDB, set_setting, get_setting
from .config import CAMERA, MODELS
from .llm_client import check_llm_server

state = SharedState()
app = FastAPI(title="RoomFlow AI")

BASE = Path(__file__).resolve().parent.parent

# Migrate settings from old settings.json if present
_settings_json = BASE / "settings.json"
if _settings_json.exists():
    try:
        with open(_settings_json) as _f:
            _old = json.load(_f)
        from .database import set_setting
        for _k in ("door_left", "door_right", "door_top", "door_bottom", "entered", "exited"):
            if _k in _old:
                set_setting(_k, str(_old[_k]))
        os.rename(_settings_json, _settings_json.with_suffix(".json.migrated"))
    except Exception:
        pass
templates = Jinja2Templates(directory=str(BASE / "templates"))

if (BASE / "static").exists():
    app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


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
    state.person_db = PersonDB()
    state.entered = state.person_db.count_events("entered")
    state.exited = state.person_db.count_events("exited")
    state.llm_status = check_llm_server()
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
    return templates.TemplateResponse(request, "person_detail.html",
                                      {"person_id": person_id})


@app.get("/settings")
async def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html", {
        "camera_ip": CAMERA["ip"],
        "camera_url": CAMERA["rtsp_url"],
        "yolo_model": MODELS["yolo"],
        "llm_model": MODELS["llm"]["model"],
        "llm_url": get_setting("llm_url", MODELS["llm"]["base_url"]),
        "vision_model": get_setting("vision_model", MODELS["vision"]["model"]),
        "entered": state.entered,
        "exited": state.exited,
        "door_left": state.door_left,
        "door_right": state.door_right,
        "door_top": state.door_top,
        "door_bottom": state.door_bottom,
        "llm_status": state.llm_status,
    })


@app.get("/video")
def video():
    return StreamingResponse(gen_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


class DoorZoneModel(BaseModel):
    door_left: float
    door_right: float
    door_top: float = 0.0
    door_bottom: float = 1.0


@app.post("/api/set_door_zone")
def api_set_door_zone(data: DoorZoneModel):
    state.set_door_zone(data.door_left, data.door_right,
                        data.door_top, data.door_bottom)
    return {"ok": True, "door_left": state.door_left,
            "door_right": state.door_right,
            "door_top": state.door_top, "door_bottom": state.door_bottom}


@app.get("/api/status")
def api_status():
    return {
        "entered": state.entered,
        "exited": state.exited,
        "current": state.current_people,
        "fps": state.fps,
        "door_left": state.door_left,
        "door_right": state.door_right,
        "door_top": state.door_top,
        "door_bottom": state.door_bottom,
    }


class CheckLLMModel(BaseModel):
    url: str


@app.post("/api/check_llm")
def api_check_llm(data: CheckLLMModel):
    result = check_llm_server(data.url)
    if result["ok"]:
        state.llm_status = result
    return result


class SettingsModel(BaseModel):
    key: str
    value: str


@app.post("/api/set_setting")
def api_set_setting(data: SettingsModel):
    set_setting(data.key, data.value)
    return {"ok": True}


@app.post("/api/reset_counters")
def api_reset_counters():
    state.entered = 0
    state.exited = 0
    set_setting("entered", "0")
    set_setting("exited", "0")
    if state.person_db is not None:
        state.person_db.clear_events()
    return {"ok": True}


@app.post("/api/reset_persons")
def api_reset_persons():
    state.person_db.reset_persons()
    return {"ok": True, "repaired": False}


@app.post("/api/dedup_persons")
def api_dedup_persons():
    persons = state.person_db.get_all_persons_for_dedup()
    has_data = any(p["ai_description"] or p["face_images"] for p in persons)
    if not has_data:
        return {"ok": False, "reason": "no descriptions or images to compare"}

    prompt = "Analyze these persons. Which ones are the SAME person?\n"
    for p in persons:
        desc = p["ai_description"][:200] if p["ai_description"] else "(no description)"
        prompt += f"ID {p['id']} — {desc}\n"
    prompt += (
        "\nReturn ONLY valid JSON: a list of groups, each group is a list of person IDs that are the same person.\n"
        'Example: [[1,3],[2,5,7]] means persons 1&3 are the same, and 2&5&7 are the same.\n'
        "If all persons are unique, return []."
    )

    try:
        url = get_setting("llm_url", MODELS["llm"]["base_url"])
        model = get_setting("vision_model", MODELS.get("vision", {}).get("model", "moondream:latest"))
        resp = requests.post(
            f"{url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "temperature": 0.1,
                "max_tokens": 500,
            },
            timeout=30,
        )
        content = resp.json()["choices"][0]["message"]["content"]
        if "[" in content:
            content = content[content.index("["):content.rindex("]") + 1]
        groups = json.loads(content)
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    merged = 0
    for group in groups:
        if len(group) < 2:
            continue
        keep = min(group)
        delete = [p for p in group if p != keep]
        state.person_db.merge_persons(keep, delete)
        merged += len(delete)
    return {"ok": True, "merged": merged, "groups": groups}


@app.get("/api/events")
def api_events():
    if state.person_db is None:
        return []
    from_db = state.person_db.get_recent_events(50)
    result = []
    for e in from_db:
        ts = e.get("timestamp", "")
        if ts:
            ts = ts.split(" ")[1][:8] if " " in ts else ts[:8]
        else:
            ts = ""
        direction = e.get("type", "")
        pid = e.get("person_id")
        name = e.get("person_name") or f"#{pid}" if pid else ""
        inn = e.get("entered_count", 0)
        out = e.get("exited_count", 0)
        if name:
            msg = f"[{ts}] {direction} {name} (IN: {inn} OUT: {out})"
        else:
            msg = f"[{ts}] {direction} (IN: {inn} OUT: {out})"
        result.append(msg)
    return result[-50:]


@app.get("/api/person_events")
def api_person_events():
    if state.person_db is None:
        return []
    from_db = state.person_db.get_recent_events(100)
    result = []
    for e in from_db:
        ts = e.get("timestamp", "")
        if ts:
            ts = ts.split(" ")[1][:8] if " " in ts else ts[:8]
        result.append({
            "time": ts,
            "person_id": e.get("person_id") or e.get("track_oid"),
            "direction": e.get("type"),
            "total_visits": e.get("entered_count", 0) if e.get("type") == "entered" else e.get("exited_count", 0),
        })
    return result[-100:]


@app.get("/api/people")
def api_people(date: str = ""):
    db = state.person_db
    if db is None:
        return {"people": []}
    return {"people": db.get_all_persons(date)}


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
    files = sorted(log_dir.rglob("*.jpg"), key=os.path.getmtime, reverse=True)
    result = []
    for f in files:
        rel = str(f.relative_to(log_dir))
        if "/crops/" not in rel:
            result.append(rel)
            if len(result) >= 100:
                break
    return result


@app.get("/screenshots/{path:path}")
def serve_screenshot(path: str):
    full = BASE / "logs" / path
    if not full.exists():
        return JSONResponse({"error": "not found"}, 404)
    return FileResponse(full, media_type="image/jpeg")


@app.get("/face_images/{path:path}")
def serve_face_image(path: str):
    full = BASE / "logs" / path
    if not full.exists():
        full = BASE / "logs" / "faces" / path
    if not full.exists():
        return JSONResponse({"error": "not found"}, 404)
    return FileResponse(full, media_type="image/jpeg")
