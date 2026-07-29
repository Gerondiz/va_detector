# Отчёт по проекту RoomFlow AI

## 1. Общая информация

**Название:** RoomFlow AI / VA Detector
**Назначение:** Real-time система детекции, трекинга и AI-идентификации людей через потолочную камеру, направленную на дверной проём. Подсчёт входящих/выходящих, AI-описание личности, веб-интерфейс.

**Репозиторий:** `https://github.com/Gerondiz/va_detector` (ветка `master`)

## 2. Архитектура

```
IP Camera (MJPEG 1280×1024)
  │
  ├── OpenCV → захват кадра
  ├── Detector (YOLOv8n + ByteTrack) → детекция + трекинг
  ├── PeopleCounter → логика зоны двери (вход/выход) → публикация в EventBus
  ├── EventBus (Pub/Sub):
  │   ├── LoggerSubscriber → SQLite + публикация AI-событий
  │   ├── AI-Worker-Thread (очередь) → LLMAnalyzer → LM Studio
  │   └── AISubscriber → PersonDB (создание/матчинг персон)
  ├── PersonDB (SQLite) → персистентность событий и людей
  └── FastAPI + Jinja2 → веб-интерфейс (5 страниц)
```

**Стек:** Python 3.10+, OpenCV, Ultralytics YOLOv8n, ByteTrack, FastAPI, SQLite (WAL), Jinja2, LM Studio (gemma4:e4b), Event Bus (Pub/Sub)

## 3. Компоненты

| Модуль | Файл | Назначение |
|--------|------|------------|
| **Сервер** | `backend/main.py` | FastAPI: роуты, MJPEG-стрим, API, статика |
| **Пайплайн** | `backend/pipeline.py` | CameraPipeline (фоновый тред), SharedState, PeopleCounter, LoggerSubscriber, AISubscriber |
| **EventBus** | `backend/event_bus.py` | Pub/Sub шина: AI-очередь (последовательный воркер) + параллельные потоки для лёгких подписчиков |
| **Детектор** | `backend/detector.py` | Обёртка над YOLO + ByteTrack (persist=True) |
| **БД** | `backend/database.py` | SQLite: persons, events, face_images, settings; WAL-режим |
| **LLM** | `backend/llm_client.py` | Одношаговый вызов LM Studio: system-контекст + image + task → JSON |
| **Конфиг** | `backend/config.py` | Камера (20.0.1.19), LLM (20.0.0.153:1234), модель (gemma4:e4b), system prompt |
| **Логгер** | `backend/logger.py` | JSONL-логгер (легаси, не используется в пайплайне) |
| **Аудио** | `backend/audio.py` | Whisper ASR (faster-whisper), отключён от пайплайна |
| **Шаблоны** | `templates/` | Jinja2: Dashboard, People, Person Detail, Settings, Logs |
| **Тесты** | `tests/test_counters.py` | 5 тестов PeopleCounter (вход, выход, cooldown, exit_frame) |
| **Тесты** | `tests/test_vision.py` | Проверка vision-способностей модели |
| **Скрипт запуска** | `run_server.sh` | Watchdog-цикл с авторестартом |

### 2.1 Event Bus — детали

```
           PeopleCounter (Publisher)
                 │
                 ▼
            EventBus.publish()
                 │
          ┌──────┴──────┐
          │              │
     Не-AI события    AI-события
    (entered/exited) (entered_ai/exited_ai)
          │              │
          ▼              ▼
    Параллельные    AI-Worker-Thread
    daemon-потоки   (queue.Queue)
    для каждого     └── последовательно:
    подписчика          AISubscriber → LLMAnalyzer
                        → PersonDB.assign/resolve
```

- **Не-AI**: каждый подписчик в своём `threading.Thread` (daemon)
- **AI**: единая блокирующая очередь, один воркер-поток — гарантирует отсутствие перегрузки LLM

### 2.2 PeopleCounter — детекция входа/выхода

```
Зона ДВЕРЬ: [door_left, door_right] × [door_top, door_bottom]

Для каждого трека:
  in_door = cx в [dx1,dx2] AND y2 в [dy1,dy2]

  was=True  → in_door=False:  ВОШЁЛ
    → pending_entry (0.3s таймер)
    → подтверждение: проверить cooldown (100px / 10s)
    → скриншот + crop → publish("entered")

  was=False → in_door=True:  ВОШЁЛ В ЗОНУ
    → сохранить exit_frame

  Трек потерян (не обнаружен):
    → pending_exit = был в зоне
    → 0.5s → publish("exited") со скриншотом
```

### 2.3 LLMAnalyzer — идентификация

```python
# Один запрос к LM Studio
model = gemma4:e4b
messages = [
  {"role": "system", "content": "Known persons: ID 1: tall man in blue jacket..."},
  {"role": "user", "content": [
    {"type": "text", "text": "Analyze the image. Compare with known people. Return JSON..."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  ]}
]
# Ответ: {"person_id": 0|int, "description": "str"}
```

- **Сервер**: LM Studio (`http://20.0.0.153:1234`)
- **Модель**: `gemma4:e4b` (8B, Q4_K_M), поддерживает vision через LM Studio
- **Таймаут**: 300s (CPU/GPU инференс)
- **Парсинг**: безопасное извлечение `{...}` через `find`/`rfind`
- **Проверка доступности**: `GET {url}/models` (LM Studio endpoint)

## 4. Ключевые возможности

- **Детекция + трекинг:** YOLOv8n + ByteTrack с persist=True
- **Зона двери:** Конфигурируемый прямоугольник через веб-интерфейс (нормализованные 0–1)
- **Анти-флап входа:** Pending-подтверждение 0.3s + cooldown по позиции (100px, 10s)
- **Анти-флап выхода:** Pending-подтверждение 0.5s после потери трека
- **Скриншоты:** Автосохранение при входе/выходе в `logs/YYYY-MM-DD/` + crop в `crops/`
- **AI-идентификация:** LM Studio + gemma4:e4b — один запрос с изображением, возврат JSON
- **Дедупликация:** LLM-анализ персон через `/api/dedup_persons`
- **Face-изображения:** Сохранение кропов при AI-матчинге
- **Персистентность:** SQLite (events, persons, face_images, settings)
- **Счётчики:** `entered_count`/`exited_count` в каждой строке events; вычисляются через `COUNT(*)` при старте
- **Проверка LLM:** Статус сервера на странице Settings (кнопка Check + авто-проверка при старте)
- **Веб-интерфейс:** 5 страниц, MJPEG-стрим, auto-polling каждую секунду
- **Watchdog:** `run_server.sh` перезапускает сервер при падении
- **Event Bus:** Pub/Sub разделение на лёгкие события (параллельно) и AI (последовательно)

## 5. База данных (SQLite)

**Файл:** `data.db` (в корне проекта, `.gitignore`)

**Таблицы:**

| Таблица | Назначение | Ключевые поля |
|---------|-----------|---------------|
| `events` | Все события входа/выхода | `id`, `type`(entered/exited), `person_id`(FK→persons), `track_oid`, `screenshot`, `crop`, `entered_count`, `exited_count`, `ai_description`, `timestamp` |
| `persons` | Уникальные люди (AI-описание) | `id`, `name`, `ai_description`, `gender`, `clothing`, `created_at`, `date` |
| `face_images` | Кропы лиц при событиях | `id`, `person_id`(FK), `filename`, `created_at` |
| `settings` | Ключ-значение (door zone, llm_url, counters) | `key`(PK), `value` |

**Миграция:** Старый `settings.json` → SQLite при старте (автоматически).

**WAL-режим** для лучшей производительности при конкурентных чтениях/записи.

## 6. API (все ручки)

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Дашборд (видео + статистика + события) |
| GET | `/settings` | Страница настроек |
| GET | `/logs` | Скриншоты |
| GET | `/people` | Список людей |
| GET | `/people/{id}` | Детали человека |
| GET | `/video` | MJPEG-стрим (multipart/x-mixed-replace) |
| GET | `/api/status` | `entered`, `exited`, `current`, `fps`, `door_*` |
| GET | `/api/events` | Последние 50 событий (текст, из БД) |
| GET | `/api/person_events` | Последние 100 событий (JSON для дашборда) |
| GET | `/api/people` | Все персоны (с фильтром `?date=`) |
| GET | `/api/people/{id}` | Детали персоны + события + face_images |
| POST | `/api/people/rename` | Переименовать персону |
| POST | `/api/set_setting` | Сохранить `{key, value}` в settings |
| POST | `/api/set_door_zone` | Сохранить `door_*` (4 числа) |
| POST | `/api/check_llm` | Проверить доступность LLM сервера |
| POST | `/api/reset_counters` | Сбросить entered/exited |
| POST | `/api/reset_persons` | Сбросить всех персон и перепривязать события |
| POST | `/api/dedup_persons` | LLM-дедупликация персон |
| GET | `/api/screenshots` | Список скриншотов (без `/crops/`) |
| GET | `/screenshots/{path}` | Отдать скриншот |
| GET | `/face_images/{path}` | Отдать face-изображение |

## 7. Конфигурация

**Файл:** `backend/config.py`

```python
CAMERA = {
    "ip": "20.0.1.19",
    "mjpg_url": "http://admin:@20.0.1.19/video/mjpg.cgi",
    "rtsp_url": "rtsp://admin@20.0.1.19:554/play1.sdp",
    "use_mjpg": True,
    "resolution": (1280, 1024),
}

MODELS = {
    "yolo": "yolov8n.pt",
    "llm": {"base_url": "http://20.0.0.153:1234", "model": "gemma4:e4b"},
    "vision": {"model": "gemma4:e4b"},
}

LLM_SYSTEM_PROMPT = """...классификация safe/threat/uncertain..."""
```

Настройки переопределяются через SQLite (`settings` table) — `llm_url`, `vision_model`, `door_*`, `entered`/`exited`.

## 8. Структура файлов проекта

```
RoomFlow AI/
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI сервер
│   ├── pipeline.py      # CameraPipeline, PeopleCounter, SharedState, подписчики
│   ├── event_bus.py     # EventBus (Pub/Sub + AI-очередь)
│   ├── detector.py      # YOLO + ByteTrack обёртка
│   ├── llm_client.py    # LLMAnalyzer + check_llm_server
│   ├── database.py      # PersonDB (SQLite)
│   ├── config.py        # Настройки камеры, LLM, промптов
│   ├── logger.py        # JSONL-логгер (легаси)
│   └── audio.py         # Whisper ASR (отключён)
├── templates/
│   ├── base.html        # Основной шаблон (навигация, стили)
│   ├── dashboard.html   # Страница с видео и событиями
│   ├── logs.html        # Просмотр скриншотов
│   ├── settings.html    # Настройки + статус LLM
│   ├── people.html      # Список людей
│   └── person_detail.html # Детали человека
├── tests/
│   ├── test_counters.py     # 5 тестов PeopleCounter
│   └── test_vision.py       # Проверка vision модели
├── Docs/
│   ├── ТЗ.md                # Техническое задание
│   ├── event_bus.md         # Документация Event Bus архитектуры
│   └── report.md            # Данный файл
├── static/              # Статические файлы
├── logs/                # Скриншоты, кропы, pipeline_error.log
├── run_server.sh        # Watchdog-скрипт запуска
├── data.db              # SQLite (в .gitignore)
└── requirements.txt
```

## 9. Data Flow (полный цикл)

### Вход человека в комнату

```
1. YOLO детектирует человека → ByteTrack присваивает oid
2. PeopleCounter.update():
   - Человек в зоне → был в зоне → вышел из зоны
   - pending_entry = True (0.3s таймер)
3. Через 0.3s:
   - Cooldown check (позиция, 100px/10s)
   - Скриншот → logs/{date}/person_{oid}_entered_{ts}.jpg
   - Crop → logs/{date}/crops/crop_oid{oid}_{ts}.jpg
   - publish("entered", {track_id, screenshot, crop, bbox})
4. LoggerSubscriber (новый поток):
   - state.entered += 1
   - set_setting("entered", str(state.entered))
   - INSERT INTO events (type='entered', entered_count, exited_count, ...)
   - publish("entered_ai", {..., event_id})
5. AI-Worker-Thread (очередь):
   - Читает crop с диска
   - existing = PersonDB.get_all_persons_with_descriptions()
   - result = LLMAnalyzer.identify_person(crop, existing)
   - Если person_id > 0: assign_event(event_id, pid)
   - Если person_id == 0: resolve_event(event_id, desc) → создаёт Person
   - add_face_image(pid, crop)
6. Web UI: /api/events (из БД), /api/status (state.entered)
```

### Выход человека из комнаты

```
1. Человек не обнаружен → трек перемещён в _recently_lost
2. Через 0.5s:
   - Скриншот из exit_frame (сохранённого при входе в зону)
   - publish("exited", {track_id, screenshot, crop, bbox, person_id})
3. Аналогично входу: LoggerSubscriber → DB → AI (если crop)
```

## 10. Тесты

| Файл | Тесты | Статус |
|------|-------|--------|
| `tests/test_counters.py` | 5 тестов: вход (0.3s delay), выход (0.5s delay), cooldown, exit_frame, screenshot | ✅ Все проходят |
| `tests/test_vision.py` | Проверка vision-способностей gemma4:e4b через LM Studio | 🧪 Ручной |

**Запуск:**
```bash
cd "RoomFlow AI" && python3 tests/test_counters.py
cd "RoomFlow AI" && python3 tests/test_vision.py [path_to_image]
```

## 11. Сильные стороны

- **Полноценный продукт:** от камеры до веб-интерфейса + AI-идентификация
- **Event Bus архитектура:** чёткое разделение Publisher (PeopleCounter) и Subscribers (логгер, AI)
- **AI-очередь:** последовательная обработка vision-запросов, не перегружает модель
- **Отказоустойчивость:** watchdog + try/except + логирование ошибок
- **Анти-флап:** pending-подтверждение для входа (0.3s) и выхода (0.5s) + cooldown по позиции
- **Runtime-конфигурация:** door zone, LLM URL — через веб-интерфейс без перезапуска
- **Скриншоты с датами:** `logs/YYYY-MM-DD/` с авто-каталогизацией
- **Счётчики из БД:** не теряются при перезапуске
- **Документация:** ТЗ, Event Bus, отчёт

## 12. Слабые места / TODO

- **Безопасность:** Пароль камеры в `config.py` (пустая строка)
- **Производительность YOLO:** ~10 FPS на CPU (1280×1024), detection каждый кадр
- **GPU отсутствует:** весь инференс на CPU (YOLO, LLM через LM Studio)
- **Аудио отключено:** `AudioProcessor` есть, не подключён к пайплайну
- **LLM-анализ безопасности:** `LLM_SYSTEM_PROMPT` есть в `config.py`, не используется в пайплайне (нет LLMAnalyzer в CameraPipeline)
- **Легаси logger.py:** JSONL-логгер определён, не используется (все события через SQLite)
- **Гонка данных:** `SharedState.entered`/`exited` без блокировки (обновляется из Subscriber-потоков)
- **Отсутствие Docker:** запуск только через `run_server.sh`
- **Переменные окружения:** пароль камеры, URL LLM жёстко в config.py
- **Нет CI/CD:** нет автоматического прогона тестов
- **Нет OpenAPI/Swagger документации:** FastAPI генерирует auto, но нет описаний ручек
- **Нет rate limiting:** API не защищён от частых запросов
- **LM Studio зависимость:** gemma4:e4b может не поддерживать vision в некоторых сборках LM Studio

## 13. Рекомендации

1. **Добавить аутентификацию** (basic auth или session) на API + веб
2. **Синхронизировать SharedState** через `threading.Lock` для `entered`/`exited`
3. **Подключить Whisper** обратно в пайплайн для audio-анализа
4. **Расширить тесты:** pytest с mock EventBus, параметризованные тесты PeopleCounter, тесты database.py
5. **Docker-образ** для деплоя (python:3.11-slim + зависимости)
6. **Переменные окружения** для чувствительных данных (пароль камеры, URL LLM)
7. **Убрать `logger.py`** или переписать поверх SQLite
8. **OpenAPI/Swagger теги** для каждой ручки
9. **Добавить LLM-анализ безопасности** в пайплайн (используя `LLM_SYSTEM_PROMPT`)
10. **Health endpoint** `/api/health` для мониторинга (camera, LLM, DB)
