import requests
import json
import threading
import base64
import cv2
import numpy as np
from config import MODELS

SYSTEM_PROMPT = """Ты — система анализа событий безопасности. Классифицируй ситуацию по описанию кадра и распознанной речи. Отвечай строго в JSON: {"classification": "safe|threat|uncertain", "reason": "...", "action": "..."}"""


class LLMAnalyzer:
    def __init__(self):
        cfg = MODELS["llm"]
        self.base_url = cfg["base_url"]
        self.model = cfg["model"]
        self.result = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _run(self, objects: list[dict], speech: str | None):
        scene = [f"{o['label']} (conf {o['confidence']})" for o in objects]
        if speech:
            scene.append(f"Speech: '{speech}'")
        user_prompt = f"Кадр: {', '.join(scene)}."

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json={"model": self.model, "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ], "temperature": 0.1},
                timeout=60,
            )
            content = resp.json()["choices"][0]["message"]["content"]
            if "{" in content:
                content = content[content.index("{"):content.rindex("}") + 1]
            result = json.loads(content)
        except Exception as e:
            result = {"classification": "uncertain", "reason": f"LLM error: {e}", "action": "check logs"}

        with self._lock:
            self.result = result
            self._thread = None

    def request(self, objects: list[dict], speech: str | None = None):
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, args=(objects, speech), daemon=True)
            self._thread.start()

    def describe_person(self, frame: np.ndarray) -> str | None:
        try:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_b64 = base64.b64encode(buffer).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{image_b64}"

            vision_model = MODELS.get("vision", {}).get("model", "moondream:latest")
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": vision_model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this person: clothing colors, objects carried, gender, approximate age. 1-2 sentences."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }],
                    "temperature": 0.1,
                    "max_tokens": 100,
                },
                timeout=30,
            )
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[vision error: {e}]"

    def get_result(self) -> dict | None:
        with self._lock:
            r = self.result
            self.result = None
            return r
