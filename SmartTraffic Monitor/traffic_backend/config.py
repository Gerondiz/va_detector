CAMERA = {
    "ip": "",
    "user": "",
    "password": "",
    "rtsp_url": "",
    "mjpg_url": "http://109.206.96.58:8080/cam_1.cgi",
    "resolution": (704, 576),
}

MODELS = {
    "yolo": "yolov8n.pt",
}

DEFAULT_LINE_POSITION = 0.5
DEFAULT_LINE_HORIZONTAL = False
DEFAULT_CONFIDENCE = 0.3
DEFAULT_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)
