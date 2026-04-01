# Конфигурация

Все настройки читаются в `src/backend/dependencies/settings.py`. Основной файл окружения лежит в корне проекта и называется `.env`.

## Базовые переменные

| Переменная | Значение по умолчанию | Что делает |
| --- | --- | --- |
| `CAMERA_SOURCE` | `0` | Индекс камеры или путь к видеофайлу |
| `LESSONS_BEGINNING` | `09:00` | Время начала занятия |
| `LESSON_START_TIME` | нет | Альтернативное имя для времени начала занятия |
| `lessons_begining` | нет | Поддерживается для обратной совместимости |

## Настройки AI-устройств

| Переменная | Значение по умолчанию | Что делает |
| --- | --- | --- |
| `AI_DEVICE` | `auto` | Общий выбор устройства для torch-based компонентов |
| `FACE_RUNTIME_BACKEND` | `auto` | `auto`, `facenet_pytorch` или `deepface` |
| `FACE_MODEL_NAME` | `Facenet512` | Имя модели для DeepFace fallback |
| `FACE_DETECTOR_BACKEND` | `retinaface` | Детектор лица для DeepFace |
| `FACE_EMBEDDING_MODEL_NAME` | `vggface2` | Предобученные веса для `facenet-pytorch` |
| `FACE_EMBEDDING_IMAGE_SIZE` | `160` | Размер выровненного лица для FaceNet |
| `FACE_EMBEDDING_MARGIN` | `0` | Отступ вокруг лица при выравнивании |
| `FACE_MIN_DETECTION_CONFIDENCE` | `0.80` | Нижняя граница уверенности face detector |

## Пороги распознавания лица

| Переменная | Значение по умолчанию | Что делает |
| --- | --- | --- |
| `FACE_DISTANCE_THRESHOLD` | `0.50` | Максимальная cosine distance для совпадения |
| `FACE_DISTANCE_MARGIN` | `0.02` | Минимальный отрыв лучшего кандидата от второго |
| `FACE_MIN_STABLE_VOTES` | `2` | Сколько совпадений подряд нужно для подтверждения |
| `FACE_VOTE_WINDOW` | `5` | Окно голосования по `track_id` |

## Пороги вовлеченности

| Переменная | Значение по умолчанию | Что делает |
| --- | --- | --- |
| `MP_MIN_DETECTION_CONFIDENCE` | `0.5` | Нижняя граница детекции для MediaPipe |
| `MP_MIN_TRACKING_CONFIDENCE` | `0.5` | Нижняя граница трекинга для MediaPipe |
| `MP_SMOOTHING_WINDOW` | `5` | Сколько значений хранить для сглаживания |
| `MP_HIGH_THRESHOLD` | `0.72` | Граница между `medium` и `high` |
| `MP_MEDIUM_THRESHOLD` | `0.45` | Граница между `low` и `medium` |

## Правила отметки посещаемости

| Переменная | Значение по умолчанию | Что делает |
| --- | --- | --- |
| `PRESENCE_CONFIRMATION_SECONDS` | `3.0` | Сколько ученик должен держаться в кадре до записи в БД |
| `ATTENDANCE_LOG_COOLDOWN_SECONDS` | `60.0` | Защита от повторной записи в течение короткого интервала |
| `ATTENDANCE_LATE_AFTER_SECONDS` | `60.0` | Допустимая задержка после начала урока |
| `STALE_TRACK_TTL_SECONDS` | `10.0` | Через сколько секунд забывается пропавший трек |

## Что проект ищет на диске

Пути для моделей захардкожены в `settings.py` и считаются от `src/backend/assets/models`:

- `yolov8n.pt`
- `yolov8n-pose.pt`
- `face_landmarker.task`
- `pose_landmarker_full.task`

Если pose-модель лежит на месте, backend вовлеченности обычно выбирает `yolo_pose`. Если `.task`-файлы не положены, MediaPipe Tasks просто не активируется.

## Рабочий `.env` для типового сценария

```env
CAMERA_SOURCE=0
AI_DEVICE=auto
FACE_RUNTIME_BACKEND=auto
FACE_DISTANCE_THRESHOLD=0.50
FACE_DISTANCE_MARGIN=0.02
FACE_MIN_STABLE_VOTES=2
MP_HIGH_THRESHOLD=0.72
MP_MEDIUM_THRESHOLD=0.45
PRESENCE_CONFIRMATION_SECONDS=3
ATTENDANCE_LATE_AFTER_SECONDS=60
LESSONS_BEGINNING=09:00
```
