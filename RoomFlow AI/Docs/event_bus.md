# Архитектура RoomFlow AI (Event Bus + Pub/Sub)

## Общая схема

```
Камера (MJPEG)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  CameraPipeline (thread)              │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ Detector │───►│ PeopleCounter │───►│  EventBus  │ │
│  │ YOLOv8n  │    │  (Publisher)  │    │  (Pub/Sub) │ │
│  └──────────┘    └──────────────┘    └──────┬───────┘ │
│                                              │         │
└──────────────────────────────────────────────┼─────────┘
                                               │
              ┌────────────────────────────────┼──────────────┐
              │                                │              │
              ▼                                ▼              │
    ┌──────────────────┐           ┌──────────────────┐       │
    │  LoggerSubscriber │           │  AI-Worker-Thread │       │
    │  (свой поток)     │           │  (очередная АИ)   │       │
    │                   │           │                   │       │
    │ • entered → DB    │           │ • entered_ai →    │       │
    │   + publish AI    │           │   LLMAnalyzer      │       │
    │ • exited → DB     │           │ • exited_ai →     │       │
    │   + publish AI    │           │   LLMAnalyzer      │       │
    └──────┬────────────┘           └────────┬──────────┘       │
           │                                 │                  │
           ▼                                 ▼                  │
    ┌──────────────────────────────────────────────┐           │
    │              SQLite (data.db)                 │           │
    │  ┌──────────┐  ┌────────┐  ┌─────────────┐   │           │
    │  │  events  │  │ persons│  │ face_images  │   │           │
    │  │ enter/exit│  │ AI desc│  │ crops        │   │           │
    │  └──────────┘  └────────┘  └─────────────┘   │           │
    └──────────────────────────────────────────────┘           │
                                                                │
    ┌────────────────────────────────────────────────────┐      │
    │              SharedState (in-memory)               │      │
    │  entered: int, exited: int, current_people: int    │      │
    │  door zone: left/right/top/bottom                  │      │
    │  last_frame, fps, llm_status                       │      │
    └────────────────────────────────────────────────────┘      │
```

## Компоненты

### 1. EventBus (`backend/event_bus.py`)

Центральная шина событий. Работает по принципу **Издатель — Подписчик**.

**Типы событий:**

| Событие | Источник | Описание |
|---------|----------|----------|
| `entered` | PeopleCounter | Человек вошёл в комнату (подтверждён через 0.3s) |
| `exited` | PeopleCounter | Человек вышел из комнаты (подтверждён через 0.5s) |
| `entered_ai` | LoggerSubscriber | Требуется AI-идентификация вошедшего |
| `exited_ai` | LoggerSubscriber | Требуется AI-идентификация вышедшего |

**Механизмы доставки:**

- **Не-AI события** (`entered`, `exited`) — для каждого подписчика создаётся **новый daemon-поток** (`SubThread-{event_type}`). Подписчики выполняются параллельно и независимо.
- **AI-события** (`entered_ai`, `exited_ai`) — ставятся в **единую блокирующую очередь** (`queue.Queue`). Единственный **AI-Worker-Thread** последовательно забирает задачи и выполняет их одну за другой. Это гарантирует, что AI-модель не перегружается параллельными запросами.

**Подписчики, зарегистрированные в `CameraPipeline.run()`:**

```python
event_bus.subscribe("entered", make_logger_subscriber(state, db, event_bus))
event_bus.subscribe("exited",  make_logger_subscriber(state, db, event_bus))
event_bus.subscribe("entered_ai", make_ai_subscriber(db, llm))
event_bus.subscribe("exited_ai",  make_ai_subscriber(db, llm))
```

### 2. PeopleCounter (`backend/pipeline.py`)

**Издатель (Publisher)** в Event Bus. Выполняется в главном потоке видеозахвата каждый кадр.

**Алгоритм детекции входа/выхода:**

```
Для каждого обнаруженного человека:
  in_door = находится ли низ bbox (y2) в зоне ДВЕРЬ

  Если трек НОВЫЙ:
    Если in_door:
      - сохранить exit_frame и exit_bbox (скриншот на случай выхода)
    Если трек был ПОТЕРЯН и найден в _recently_lost (<2s):
      - восстановить person_id, exit_frame, exit_bbox

  Если трек СУЩЕСТВУЕТ:
    was = предыдущее состояние in_door

    was=True  и not in_door → ВОШЁЛ:
      - Установить pending_entry (таймер 0.3s)
      - Сохранить entry_frame и entry_bbox

    was=False и in_door     → ВОШЁЛ В ЗОНУ:
      - Обновить exit_frame и exit_bbox (может выйти)

  В конце кадра:
    - Подтвердить pending_entry (через 0.3s если всё ещё вне зоны):
      • Проверка cooldown (100px / 10s) — защита от двойного счёта
      • Сохранить скриншот в logs/{date}/
      • Вырезать crop в logs/{date}/crops/
      • Опубликовать "entered" в EventBus

    - Потерянные треки (не обнаружены в текущем кадре):
      • Переместить в _recently_lost
      • pending_exit = был ли человек в зоне
      • Через 0.5s → сохранить скриншот, crop, опубликовать "exited"

    - Старые записи _recently_lost удаляются через 5s
```

**Зона ДВЕРЬ** задаётся пропорциями от кадра:
```python
dx1 = w * state.door_left   # 0.0 – 1.0
dx2 = w * state.door_right
dy1 = h * state.door_top
dy2 = h * state.door_bottom
in_door = dx1 <= cx <= dx2 and dy1 <= y2 <= dy2
```

### 3. LoggerSubscriber (`backend/pipeline.py:make_logger_subscriber`)

Выполняется в отдельном daemon-потоке для каждого события.

```python
entered → state.entered += 1
       → set_setting("entered", str(state.entered))
       → db.save_entry_event(track_id, crop, screenshot, entered_count, exited_count)
       → event_bus.publish("entered_ai", {..., event_id})

exited  → state.exited += 1
       → set_setting("exited", str(state.exited))
       → db.save_exit_event(...)
       → event_bus.publish("exited_ai", {..., event_id})  # только если есть crop
```

### 4. AISubscriber (`backend/pipeline.py:make_ai_subscriber`)

Выполняется **только** в AI-Worker-Thread, строго последовательно.

```python
entered_ai / exited_ai:
  - Проверить что event_id и crop_path существуют
  - Проверить что событие ещё не привязано к person_id
  - Загрузить crop с диска
  - Получить список известных людей: db.get_all_persons_with_descriptions()
  - Вызвать llm.identify_person(crop, existing)
  - Если result.person_id > 0:
      db.assign_event(event_id, pid)
      db.add_face_image(pid, crop)       # сохранить кроп как лицо
  - Если result.person_id == 0:
      pid = db.resolve_event(event_id, desc)  # создать нового Person
      db.add_face_image(pid, crop)
```

### 5. LLMAnalyzer (`backend/llm_client.py`)

**Одношаговый подход** — один запрос к LM Studio, который возвращает JSON.

```python
system: "You are a security AI. Context: {список известных людей с описаниями}"
user:   [
  {"type": "text", "text": "Analyze the image. Describe clothing in 1 sentence.
                            Compare with known people. Return JSON:
                            {\"person_id\": <int>, \"description\": \"<str>\"}"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
]
```

- **Сервер**: LM Studio (`http://20.0.0.153:1234`)
- **Модель**: `gemma4:e4b` (8B, text + vision через LM Studio)
- **Таймаут**: 300s (на случай холодного старта / CPU-инференса)
- **Парсинг ответа**: безопасное извлечение `{...}`, try/except

**Проверка сервера** (`check_llm_server`):
- GET `{url}/models` (эндпоинт LM Studio)
- Возвращает `{"ok": True/False, "models": [...], "choices": [...]}`

### 6. Database (`backend/database.py`)

**SQLite** с WAL-режимом, таблицы:

| Таблица | Назначение | Ключевые поля |
|---------|-----------|---------------|
| `events` | Все входы/выходы | `id`, `type`(entered/exited), `person_id`, `track_oid`, `screenshot`, `crop`, `entered_count`, `exited_count`, `timestamp` |
| `persons` | Люди (AI-ID) | `id`, `name`, `ai_description`, `gender`, `clothing` |
| `face_images` | Кропы лиц | `id`, `person_id`, `filename` |
| `settings` | Ключ-значение | `key`, `value` — door zone, llm_url, counters |

**Счётчики** вычисляются через `SELECT COUNT(*)` при старте, а не из `settings`:

```python
state.entered = db.count_events("entered")
state.exited  = db.count_events("exited")
```

## Data Flow (полный цикл)

### Вход в комнату

```
1. YOLO детектирует человека, трекер ByteTrack присваивает ID
2. PeopleCounter: was=True (был в зоне), in_door=False (вышел)
   → pending_entry = True, сохранён entry_frame
3. Через 3–10 кадров (0.3s):
   → подтверждение: человек всё ещё вне зоны
   → cooldown check (100px / 10s)
   → сохранение скриншота и кропа на диск
   → publish("entered", {track_id, screenshot, crop, bbox})
4. LoggerSubscriber (новый поток):
   → state.entered += 1
   → set_setting("entered", str(state.entered))
   → INSERT INTO events (type='entered', ...)
   → publish("entered_ai", {..., event_id})
5. AI-Worker-Thread (очередь):
   → загружает crop с диска
   → llm.identify_person(crop, existing_persons)
   → если найден → db.assign_event(event_id, pid)
   → если новый → db.resolve_event(event_id, desc) → создаёт Person
   → db.add_face_image(pid, crop)
6. Web UI обновляется через /api/events (из БД) и /api/person_events
```

### Выход из комнаты

```
1. PeopleCounter: трек потерян (person не обнаружен)
   → перемещение в _recently_lost с pending_exit=True
2. Через 0.5s (5–10 кадров):
   → подтверждение выхода
   → publish("exited", {track_id, screenshot, crop, bbox, person_id})
3-5. Аналогично входу (LoggerSubscriber → DB → AI если есть crop)
```

## SharedState (`backend/pipeline.py`)

Общее состояние, доступное из FastAPI и Pipeline:

```python
@dataclass
class SharedState:
    frame:           np.ndarray       # последний аннотированный кадр
    entered:         int              # счётчик входов (из COUNT(*))
    exited:          int              # счётчик выходов
    current_people:  int              # людей сейчас (всегда 0, может быть удалено)
    running:         bool             # флаг работы
    door_left/right/top/bottom: float # зона двери (0-1)
    fps:             int              # FPS последней секунды
    person_db:       PersonDB | None
    last_frame_time: float
    llm_status:      dict             # результат check_llm_server()
```

## FastAPI Routes (`backend/main.py`)

| Route | Метод | Описание |
|-------|-------|----------|
| `/` | GET | Дашборд (MJPEG + события) |
| `/settings` | GET | Страница настроек |
| `/logs` | GET | Скриншоты |
| `/people` | GET | Люди |
| `/people/{id}` | GET | Детали человека |
| `/video` | GET | MJPEG-стрим |
| `/api/status` | GET | entered/exited/fps/door |
| `/api/events` | GET | Последние 50 событий (из БД) |
| `/api/person_events` | GET | Последние 100 для дашборда |
| `/api/people` | GET | Список людей |
| `/api/people/{id}` | GET | Детали человека |
| `/api/people/rename` | POST | Переименовать |
| `/api/set_setting` | POST | Сохранить настройку |
| `/api/set_door_zone` | POST | Сохранить зону двери |
| `/api/check_llm` | POST | Проверить LLM сервер |
| `/api/reset_counters` | POST | Сбросить счётчики |
| `/api/reset_persons` | POST | Сбросить людей |
| `/api/dedup_persons` | POST | AI-дедупликация |
| `/api/screenshots` | GET | Список скриншотов |

## Startup Flow

```
uvicorn main:app
  │
  ├── FastAPI startup event
  │   ├── PersonDB() → init_db() → CREATE TABLE IF NOT EXISTS ...
  │   ├── state.entered = db.count_events("entered")
  │   ├── state.exited  = db.count_events("exited")
  │   ├── state.llm_status = check_llm_server()
  │   └── CameraPipeline(state).start()
  │         │
  │         ├── CameraPipeline.run() (новый поток)
  │         │   ├── cap = VideoCapture(CAMERA["mjpg_url"])
  │         │   ├── Detector() (YOLO + ByteTrack)
  │         │   ├── EventBus() → AI-Worker-Thread стартует
  │         │   ├── LLMAnalyzer()
  │         │   ├── PeopleCounter(state, event_bus)
  │         │   └── Подписка: entered/exited → LoggerSubscriber
  │         │                entered_ai/exited_ai → AISubscriber
  │         │
  │         └── Цикл захвата:
  │             cap.read() → detect() → counter.update() → publish events
  │
  └── FastAPI shutdown event
      └── state.running = False → pipeline завершается
```

## Зависимости

- **YOLOv8n** — детекция людей (Ultralytics)
- **ByteTrack** — трекинг (встроен в Ultralytics)
- **LM Studio** (`http://20.0.0.153:1234`) — LLM + vision инференс
- **Модель**: `gemma4:e4b` (8B, Q4_K_M)
- **SQLite** — долговременное хранение событий и людей
- **FastAPI** + **Jinja2** — веб-интерфейс
