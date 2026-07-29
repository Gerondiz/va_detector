import os
import sys
import base64
import json
import requests

# =====================================================================
# НАСТРОЙКИ ДЛЯ LM STUDIO
# =====================================================================
# Укажите IP-адрес вашей основной машины, где запущена LM Studio
LM_STUDIO_HOST = "20.0.0.153"
# Укажите порт, который вы запустили в LM Studio (по умолчанию 1234, либо 11434)
LM_STUDIO_PORT = "1234" 

# Точное имя модели из верхней строки интерфейса LM Studio (например, "gemma4:e4b")
# Важно: имя должно совпадать с тем, что написано в LM Studio!
MODEL_NAME = "gemma4:e4b" 

IMAGE_PATH = "test.jpg"
# =====================================================================

LM_STUDIO_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/chat/completions"

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Ошибка: Положите тестовую картинку рядом со скриптом и назовите её '{IMAGE_PATH}'")
    sys.exit(1)

# 1. Читаем картинку и кодируем в чистый Base64 Data URL (стандарт OpenAI для LM Studio)
with open(IMAGE_PATH, "rb") as image_file:
    raw_b64 = base64.b64encode(image_file.read()).decode("utf-8")
    clean_b64 = raw_b64.replace("\n", "").replace("\r", "").strip()
    data_url = f"data:image/jpeg;base64,{clean_b64}"

# 2. Имитируем базу данных из 5 человек, чтобы проверить логику матчинга
mock_database_context = (
    "List of already known people in the room:\n"
    "Person ID 1: Man in a black hoodie and dark sweatpants\n"
    "Person ID 2: Woman in a white t-shirt and blue jeans\n"
    "Person ID 3: Person wearing a red jacket and a grey cap\n"
    "Person ID 4: Individual in a blue business suit and brown shoes\n"
    "Person ID 5: Courier in a yellow delivery jacket with a large backpack\n"
)

# 3. Промпт задачи для зрения
task_prompt = (
    "Analyze the attached image of the person who just walked into the room.\n"
    "1. Describe their clothing and appearance in 1 short sentence in English.\n"
    "2. Compare them with the 'List of already known people' provided in the system context message.\n"
    "3. If they match any known person, output their existing ID. If they are new, output ID 0.\n\n"
    "You MUST reply ONLY with a raw JSON object. Do not wrap it in markdown code blocks. Format exactly:\n"
    '{"person_id": <int_id>, "description": "<your_short_clothing_description>"}'
)

# 4. Упаковываем сообщения по стандарту OpenAI Vision API
messages = [
    {
        "role": "system",
        "content": f"You are a security AI system. Here is the context of your room database:\n{mock_database_context}"
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": task_prompt},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]
    }
]

print("==================================================")
print(f" 🚀 ТЕСТИРОВАНИЕ VISION ЧЕРЕЗ СЕРВЕР LM STUDIO")
print("==================================================")
print(f"📍 Эндпоинт: {LM_STUDIO_URL}")
print(f"🤖 Модель: {MODEL_NAME}")
print("⏳ Отправка запроса (таймаут 5 минут)...")

try:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2
    }
    
    resp = requests.post(LM_STUDIO_URL, json=payload, timeout=300)
    
    if resp.status_code == 200:
        print("\n✅ Ответ от LM Studio успешно получен!")
        resp_data = resp.json()
        
        # Безопасно извлекаем текст из стандартной структуры OpenAI
        try:
            if "choices" in resp_data and len(resp_data["choices"]) > 0:
                raw_output = resp_data["choices"][0]["message"]["content"].strip()
                print("\n--- ОТВЕТ МОДЕЛИ ---")
                print(raw_output)
                print("--------------------")
            else:
                print("\n⚠️ В JSON отсутствует массив 'choices'. Ответ сервера:")
                print(json.dumps(resp_data, indent=2, ensure_ascii=False))
        except Exception as parse_err:
            print(f"\n❌ Не удалось распарсить текст ответа: {parse_err}")
            print(json.dumps(resp_data, indent=2, ensure_ascii=False))
            
    else:
        print(f"\n❌ Ошибка сервера LM Studio: Код {resp.status_code}")
        print(resp.text)

except Exception as e:
    print(f"\n❌ Критический сбой сети: {e}")
    print("💡 Проверьте, включен ли сервер в LM Studio и доступен ли порт снаружи ВМ.")

print("\n==================================================")
