import requests
import json
import base64
import cv2
import numpy as np
from .config import MODELS
from .database import get_setting


class LLMAnalyzer:
    def __init__(self):
        self._base_url: str | None = None
        self._vision_model: str | None = None

    def _get_url(self) -> str:
        if self._base_url is None:
            self._base_url = get_setting("llm_url", MODELS["llm"]["base_url"])
        return self._base_url

    def _get_vision_model(self) -> str:
        if self._vision_model is None:
            self._vision_model = get_setting("vision_model", MODELS.get("vision", {}).get("model", "moondream:latest"))
        return self._vision_model

    def identify_person(self, crop: np.ndarray, existing: list[dict]) -> dict:
        try:
            _, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_b64 = base64.b64encode(buffer).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{image_b64}"

            persons_text = "Known persons:\n"
            if not existing:
                persons_text += "(none yet)\n"
            else:
                for p in existing:
                    desc = p.get("ai_description", "").strip()
                    if desc:
                        persons_text += f"ID {p['id']}: {desc[:200]}\n"

            prompt = (
                "You are a person identification system.\n"
                f"{persons_text}\n"
                "Analyze the person in the image. If they match a known person, return that person's ID. "
                "If they are new, return ID 0.\n"
                "Also describe this person: clothing colors, objects carried, gender, approximate age (1-2 sentences).\n"
                "Return ONLY valid JSON, no extra text:\n"
                '{"person_id": <int>, "description": "<str>"}'
            )

            resp = requests.post(
                f"{self._get_url()}/chat/completions",
                json={
                    "model": self._get_vision_model(),
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }],
                    "temperature": 0.1,
                    "max_tokens": 200,
                },
                timeout=30,
            )
            content = resp.json()["choices"][0]["message"]["content"]
            if "{" in content:
                content = content[content.index("{"):content.rindex("}") + 1]
            return json.loads(content)
        except Exception as e:
            return {"person_id": 0, "description": f"[error: {e}]"}
