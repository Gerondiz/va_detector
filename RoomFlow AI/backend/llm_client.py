import requests
import json
import base64
import cv2
import numpy as np
import time
import traceback
from .config import MODELS
from .database import get_setting


def _ollama_base(url: str) -> str:
    return url.rsplit("/v1", 1)[0] if url.endswith("/v1") else url


def check_llm_server(base_url: str | None = None) -> dict:
    url = base_url or get_setting("llm_url", MODELS["llm"]["base_url"])
    try:
        resp = requests.get(f"{_ollama_base(url)}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class LLMAnalyzer:
    def _get_url(self) -> str:
        return get_setting("llm_url", MODELS["llm"]["base_url"])

    def _get_vision_model(self) -> str:
        return get_setting("vision_model", MODELS.get("vision", {}).get("model", "moondream:latest"))

    def _get_llm_model(self) -> str:
        return MODELS["llm"]["model"]

    def _parse_json(self, text: str) -> dict | None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end >= start:
            return json.loads(text[start:end + 1])
        return None

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

            model = self._get_llm_model()
            resp = requests.post(
                f"{self._get_url()}/chat/completions",
                json={
                    "model": model,
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
                timeout=60,
            )
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if not content:
                return {"person_id": 0, "description": "[empty response]"}

            parsed = self._parse_json(content)
            if parsed is not None:
                return parsed

            return {"person_id": 0, "description": f"[bad format: {content[:200]}]"}

        except Exception as e:
            with open("logs/pipeline_error.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} LLMAnalyzer: {e}\n")
                traceback.print_exc(file=f)
            return {"person_id": 0, "description": f"[error: {e}]"}
