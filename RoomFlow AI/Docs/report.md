# Отчёт по проекту RoomFlow AI

## 1. Общая информация

**Название:** RoomFlow AI / VA Detector
**Назначение:** Система real-time детекции, трекинга и идентификации людей через потолочную камеру, направленную на дверной проём. Подсчёт входящих/выходящих, AI-описание личности, веб-интерфейс.

## 2. Архитектура

```
IP Camera (MJPEG/RTSP)
  │
  ├── OpenCV → захват кадра
  ├── YOLOv8 → детекция person
  ├── ByteTrack → трекинг (ID)
  ├── PeopleCounter → логика зоны двери (вход/выход)
  ├── LLMAnalyzer → vision-модель (moondream) → описание/ID
  ├── PersonDB (SQLite) → персистентность
  └── FastAPI + Jinja2 → веб-интерфейс
```

**Стек:** Python 3.10+, OpenCV, Ultralytics YOLOv8, ByteTrack, FastAPI, SQLite, Jinja2, Ollama (moondream)

## 3. Компоненты

| Модуль | Файл | Назначение |
|--------|------|------------|
| **Сервер** | `backend/main.py` | FastAPI-приложение: роуты, MJPEG-стрим, API |
| **Пайплайн** | `backend/pipeline.py` | CameraPipeline (фоновый тред), SharedState, PeopleCounter (логика входа/выхода) |
| **Детектор** | `backend/detector.py` | Обёртка над YOLO + ByteTrack |
| **БД** | `backend/database.py` | SQLite: persons, events, face_images, settings |
| **LLM** | `backend/llm_client.py` | Vision-клиент к Ollama (moondream) для описания/матчинга людей |
| **Конфиг** | `backend/config.py` | Настройки камеры, путей, LLM |
| **Логгер** | `backend/logger.py` | JSONL-логирование событий (легаси) |
| **Аудио** | `backend/audio.py` | Whisper ASR через ffmpeg + RTSP (отключён в пайплайне) |
| **Шаблоны** | `templates/` | Jinja2: Dashboard, People, Person Detail, Settings, Logs |
| **Скрипт запуска** | `run_server.sh` | Watchdog-цикл с авторестартом |

## 4. Ключевые возможности

- **Детекция + трекинг:** YOLOv8n + ByteTrack с периодическим сбросом трекера (каждые 1500 кадров)
- **Зона двери:** Конфигурируемый прямоугольник (нормализованные 0-1); вход фиксируется при пересечении нижней границы зоны (координата `y2` ног), выход — при исчезновении объекта из кадра с задержкой 0.5с
- **Анти-флап:** Cooldown по позиции (100px, 10с) для предотвращения ложных срабатываний
- **AI-идентификация:** Vision LLM (moondream) описывает внешность человека; матчинг через Jaccard similarity по словам описания (порог 0.55)
- **Дедупликация:** LLM анализирует список персон и группирует дубликаты (через `/api/dedup_persons`)
- **Face-изображения:** Автосохранение кропов лиц при событиях входа
- **Персистентность:** SQLite (persons, events, face_images, settings)
- **Веб-интерфейс:** 5 страниц (Dashboard, People, Person Detail, Settings, Logs), MJPEG-стрим, auto-polling
- **Watchdog:** `run_server.sh` перезапускает сервер при падении (известная проблема OpenCV MJPEG)

## 5. База данных (SQLite)

- **persons:** id, name, gender, clothing, ai_description, created_at, date
- **events:** id, person_id (FK), type (entered/exited), timestamp, screenshot, crop, ai_description, track_oid
- **face_images:** id, person_id (FK), filename, created_at
- **settings:** key/value таблица

Миграция из старого `settings.json` → SQLite при старте.

## 6. API (основные ручки)

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/status` | entered, exited, current, fps, door zone |
| GET | `/api/events` | Текстовые события (in-memory + DB) |
| GET | `/api/people` | Список персон (с фильтром по дате) |
| GET | `/api/people/{id}` | Детали персоны + события |
| POST | `/api/people/rename` | Переименовать персону |
| POST | `/api/reset_counters` | Сброс entered/exited |
| POST | `/api/set_door_zone` | Обновить зону двери |
| POST | `/api/set_setting` | Сохранить настройку |
| POST | `/api/reset_persons` | Сбросить всех персон |
| POST | `/api/dedup_persons` | LLM-дедупликация персон |
| GET | `/video` | MJPEG-стрим |

## 7. Сильные стороны

- **Полноценный продукт:** Работает от камеры до веб-интерфейса
- **Отказоустойчивость:** watchdog + обработка ошибок + логирование в файл
- **Конфигурация в runtime:** door zone, LLM URL, vision model — через веб-интерфейс
- **AI-матчинг:** Vision LLM для кросс-сессионной идентификации
- **Дедупликация:** LLM автоматически находит дубликаты персон
- **Тёмная тема UI:** Современный дизайн

## 8. Слабые места / TODO

- **Безопасность:** Пароль от камеры жёстко закодирован в `config.py` (пустая строка, но всё равно)
- **Производительность:** YOLOv8n обрабатывается каждый 2-й кадр, LLM-запросы в отдельном треде — но при высокой нагрузке очередь `_ai_queue` может расти
- **Отсутствие тестов:** Нет юнит/интеграционных тестов
- **Аудио отключено:** `AudioProcessor` определён, не подключен к пайплайну (комментарий "gemma4 text-only, no vision")
- **LLM-анализ безопасности:** Код для промпта безопасности есть в `config.py`, но закомментирован в пайплайне (строка 371)
- **Отсутствие type hints:** Только базовые, многие функции без аннотаций
- **Устаревший logger:** `EventLogger` пишет JSONL, но почти не используется (события идут через SQLite + SharedState)
- **Гонка данных:** SharedState использует блокировку только для `get_frame`/`update_frame`, но `person_visits`, `recent_events` и др. не синхронизированы
- **Жёсткие пути:** `logs/` и `data.db` хардкодятся относительно папки проекта
- **Нет Docker/контейнеризации:** Запуск только через `run_server.sh`

## 9. Рекомендации

1. **Добавить аутентификацию** (хотя бы basic auth) на API/веб
2. **Переписать SharedState** с полноценной синхронизацией через `threading.RLock`
3. **Подключить Whisper** обратно в пайплайн (аудио-анализ)
4. **Добавить тесты** (pytest) на core-логику (PeopleCounter, Database)
5. **Docker-образ** для быстрого деплоя
6. **Переменные окружения** для чувствительных данных (пароль камеры, ключи)
7. **Убрать устаревший `logger.py`** или переписать его поверх БД
8. **Реализовать API документацию** (OpenAPI/Swagger — FastAPI делает это автоматически, нужно лишь добавить теги/описания)
