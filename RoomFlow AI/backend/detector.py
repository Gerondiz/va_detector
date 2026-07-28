from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame, conf: float = 0.5, iou: float = 0.45, track: bool = True):
        if track:
            results = self.model.track(frame, persist=True, conf=conf, iou=iou, verbose=False)
        else:
            results = self.model(frame, conf=conf, iou=iou, verbose=False)
        return results

    def get_objects(self, results) -> list[dict]:
        objects = []
        if results[0].boxes is None:
            return objects
        boxes = results[0].boxes
        for box in boxes:
            raw_id = box.id
            if raw_id is not None and raw_id.numel() > 0:
                tid = int(raw_id[0].item())
            else:
                tid = None
            obj = {
                "id": tid,
                "class": int(box.cls.item()),
                "label": results[0].names[int(box.cls.item())],
                "confidence": round(box.conf.item(), 3),
                "bbox": [round(x.item(), 1) for x in box.xyxy[0]],
            }
            objects.append(obj)

        return objects

    def get_annotated_frame(self, results):
        return results[0].plot()
