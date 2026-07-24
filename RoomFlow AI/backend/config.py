# Camera and app configuration

CAMERA = {
    "ip": "20.0.1.19",
    "user": "admin",
    "password": "",
    "rtsp_url": "rtsp://admin@20.0.1.19:554/play1.sdp",
    "mjpg_url": "http://admin:@20.0.1.19/video/mjpg.cgi",
    "use_mjpg": True,
    "resolution": (1280, 1024),
    "fps": 30,
}

MODELS = {
    "yolo": "yolov8n.pt",
    "whisper": "base",
    "llm": {
        "provider": "ollama",
        "base_url": "http://20.0.0.150:11434/v1",
        "model": "gemma4:latest",
    },
    "vision": {
        "model": "moondream:latest",
    },
}

LOGS = {
    "dir": "logs",
    "save_screenshots": True,
}

LLM_SYSTEM_PROMPT = """Ты — система анализа событий безопасности.
Классифицируй ситуацию по описанию кадра и распознанной речи.
Ответь строго в JSON:
{"classification": "safe|threat|uncertain", "reason": "...", "action": "..."}"""
