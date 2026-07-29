import sys
import threading
import queue
import traceback
import logging

# Настройка логирования для отслеживания работы шины в фоне
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EventBus")

class EventBus:
    def __init__(self):
        # Словарь подписчиков: { event_type: [callback_functions] }
        self._subscribers = {}
        
        # Потокобезопасная очередь строго для тяжелых задач ИИ
        self._ai_queue = queue.Queue()
        
        # Список типов событий, которые требуют последовательной AI-обработки
        self._ai_event_types = {"entered_ai", "exited_ai"}
        
        # Запускаем один выделенный фоновый поток для разбора AI-очереди
        self._ai_worker = threading.Thread(
            target=self._ai_worker_loop, 
            name="AI-Worker-Thread", 
            daemon=True
        )
        self._ai_worker.start()
        logger.info("Шина событий успешно инициализирована. Запущен AI-Worker.")

    def subscribe(self, event_type: str, callback):
        """Регистрация нового подписчика на определенный тип события"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.info(f"Подписчик {callback.__name__} успешно добавлен на событие '{event_type}'")

    def publish(self, event_type: str, data: dict):
        """Публикация события в шину"""
        if event_type not in self._subscribers or not self._subscribers[event_type]:
            return

        # ЕСЛИ СОБЫТИЕ ДЛЯ ИИ -> отправляем его в последовательную очередь
        if event_type in self._ai_event_types:
            logger.info(f"Тяжелое событие '{event_type}' добавлено в AI-очередь (Текущий размер: {self._ai_queue.qsize() + 1})")
            self._ai_queue.put((event_type, data))
            return

        # ДЛЯ ВСЕХ ОСТАЛЬНЫХ СОБЫТИЙ (логи, счетчики) -> запускаем мгновенный параллельный поток
        for callback in self._subscribers[event_type]:
            thread = threading.Thread(
                target=self._safe_execute,
                args=(callback, event_type, data),
                name=f"SubThread-{event_type}",
                daemon=True
            )
            thread.start()

    def _safe_execute(self, callback, event_type: str, data: dict):
        """Безопасное выполнение подписчика с перехватом ошибок, чтобы не падал основной процесс"""
        try:
            callback(event_type, data)
        except Exception as e:
            logger.error(f"Критическая ошибка в подписчике {callback.__name__} при обработке '{event_type}': {e}")
            # Логируем полный стек ошибки для отладки ИИ
            traceback.print_exc(file=sys.stderr)

    def _ai_worker_loop(self):
        """Бесконечный цикл единственного AI-потока.
        Последовательно забирает задачи из очереди и выполняет их одна за другой."""
        while True:
            try:
                # Извлекаем задачу (блокирует поток, если очередь пуста, не нагружая CPU)
                event_type, data = self._ai_queue.get()
                
                logger.info(f"AI-Worker начал обработку события '{event_type}' для track_id {data.get('track_id')}")
                
                # Вызываем всех подписчиков, которые подписаны на это AI-событие
                for callback in self._subscribers.get(event_type, []):
                    # Выполняем строго последовательно в контексте ЭТОГО ЖЕ рабочего потока
                    self._safe_execute(callback, event_type, data)
                    
                logger.info(f"AI-Worker успешно завершил обработку события '{event_type}'")
                
                # Сообщаем очереди, что задача полностью выполнена
                self._ai_queue.task_done()
                
            except Exception as e:
                logger.critical(f"Глобальный сбой в цикле AI-Worker: {e}")
                traceback.print_exc(file=sys.stderr)
