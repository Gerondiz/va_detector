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
    """Вспомогательная функция, сохранена для совместимости с FastAPI"""
    return url.rsplit("/v1", 1) if url.endswith("/v1") else url

def check_llm_server(base_url: str | None = None) -> dict:
    """Универсальная проверка доступности сервера LM Studio.
    Исправлено: возвращает структуру, которую корректно переварят и FastAPI, и бэкенд."""
    url = base_url or get_setting("llm_url", MODELS["llm"]["base_url"])
    try:
        # Пингуем стандартный эндпоинт моделей LM Studio
        resp = requests.get(f"{url.rstrip('/')}/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        # Собираем список доступных моделей
        models_list = []
        if isinstance(data, dict) and "data" in data:
            models_list = [m["id"] for m in data["data"] if "id" in m]
            
        # Возвращаем гибридный объект: он ведет себя и как словарь, и имеет поле choices для старого кода
        result = {
            "ok": True, 
            "models": models_list,
            "choices": [{"message": {"content": "connected"}}] # Заглушка для старого парсера FastAPI
        }
        return result
    except Exception as e:
        return {
            "ok": False, 
            "error": str(e),
            "choices": [{"message": {"content": f"error: {e}"}}]
        }


class LLMAnalyzer:
    def __init__(self, config_models: dict = None):
        pass

    def _get_url(self) -> str:
        """Получает актуальный URL LM Studio из базы данных на лету"""
        return get_setting("llm_url", MODELS["llm"]["base_url"])

    def _get_llm_model(self) -> str:
        """Получает имя модели из конфига"""
        return MODELS["llm"]["model"]

    def _parse_json(self, text: str) -> dict | None:
        """Безопасно извлекает JSON-структуру из текстового ответа"""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end >= start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
        return None

    def identify_person(self, crop: np.ndarray, existing: list[dict]) -> dict:
        """ОДНОШАГОВЫЙ АНАЛИЗ ДЛЯ LM STUDIO: Передает контекст в system, 
        а картинку в user. Исправлен парсинг ответа под чистый стандарт OpenAI."""
        try:
            # 1. Сжатие и кодирование кропа в стандартный OpenAI Data URL
            _, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_b64 = base64.b64encode(buffer).decode("utf-8")
            clean_b64 = image_b64.replace("\n", "").replace("\r", "").strip()
            data_url = f"data:image/jpeg;base64,{clean_b64}"

            # 2. Формируем контекст базы данных известных людей
            persons_text = "List of already known people in the room:\n"
            if not existing:
                persons_text += "(No known people yet, database is empty)\n"
            else:
                for p in existing:
                    desc = p.get("ai_description", "").strip()
                    if desc:
                        persons_text += f"Person ID {p['id']}: {desc}\n"

            # 3. Инструкция для анализа
            task_prompt = (
                "Analyze the attached image of the person who just walked into the room.\n"
                "1. Describe their clothing and appearance in 1 short sentence in English.\n"
                "2. Compare them with the 'List of already known people' provided in the system context message.\n"
                "3. If they match any known person, output their existing ID. If they are new, output ID 0.\n\n"
                "You MUST reply ONLY with a raw JSON object. Do not wrap it in markdown code blocks. Format exactly:\n"
                '{"person_id": <int_id>, "description": "<your_short_clothing_description>"}'
            )

            # 4. Формируем раздельные сообщения (Архитектура под LM Studio)
            messages = [
                {
                    "role": "system",
                    "content": f"You are a security AI system. Here is the context of your room database:\n{persons_text}"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]

            # 5. Сетевой запрос к серверу LM Studio
            resp = requests.post(
                f"{self._get_url().rstrip('/')}/chat/completions",
                json={
                    "model": self._get_llm_model(),
                    "messages": messages,
                    "temperature": 0.2
                },
                timeout=300  # Надежный таймаут для CPU/GPU инференса
            )
            resp.raise_for_status()
            
            # --- ИСПРАВЛЕННЫЙ БЛОК ЧТЕНИЯ (СТРОГО ПО СТАНДАРТУ OPENAI / LM STUDIO) ---
            resp_data = resp.json()
            result_text = resp_data["choices"][0]["message"]["content"].strip()
            
            parsed = self._parse_json(result_text)

            if parsed is not None and isinstance(parsed, dict):
                return {
                    "person_id": int(parsed.get("person_id", 0)),
                    "description": str(parsed.get("description", "[No description generated]"))
                }
            
            return {"person_id": 0, "description": f"[Failed to parse JSON. Raw response: {result_text[:100]}]"}

        except Exception as e:
            with open("logs/pipeline_error.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} LLMAnalyzer LMStudio Error: {e}\n")
                traceback.print_exc(file=f)
            return {"person_id": 0, "description": f"[error: {e}]"}
