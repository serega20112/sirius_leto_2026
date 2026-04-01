# Хранилище и SQLite

Проект хранит данные в локальных файлах и SQLite. Никакой внешней БД здесь не требуется.

## Где лежат данные

- база: `src/backend/assets/database/attendance.db`
- фотографии учеников: `src/backend/assets/images`
- модели: `src/backend/assets/models`

## Таблица `students`

Таблица создается в `src/backend/infrastructure/database.py` через `StudentModel`.

Поля:

- `id`
- `name`
- `group_name`
- `photo1_path`
- `photo2_path`
- `photo3_path`
- `created_at`

Почему три отдельных поля, а не отдельная таблица фотографий: проект изначально заточен под строго три фото на регистрацию. Для текущего масштаба это грубовато, но честно и прозрачно.

## Таблица `attendance_logs`

Таблица создается там же через `AttendanceModel`.

Поля:

- `id`
- `student_id`
- `timestamp`
- `is_late`
- `engagement_score`

`engagement_score` хранится строкой. Это не числовой confidence, а одно из значений:

- `high`
- `medium`
- `low`
- `unknown`

## Как данные проходят через слой persistence

SQLite-репозитории живут в:

- `src/backend/infrastructure/persistence/sqlite/student_repository.py`
- `src/backend/infrastructure/persistence/sqlite/attendance_repository.py`

Они решают только две задачи:

- перевести SQLAlchemy-модель в доменную сущность;
- выполнить нужный запрос.

Никакой логики про опоздание, посещаемость или AI там нет.

## Фотографии учеников

При регистрации use case создает `student_id` и сохраняет три файла под именами:

- `<student_id>_1.jpg`
- `<student_id>_2.jpg`
- `<student_id>_3.jpg`

Именно по этим именам recognizer потом собирает локальную face gallery.

## Модели

В `assets/models` проект ожидает минимум:

- `yolov8n.pt`
- `yolov8n-pose.pt`

Дополнительно можно положить:

- `face_landmarker.task`
- `pose_landmarker_full.task`

Если `.task`-файлов нет, это не авария. Просто MediaPipe Tasks не будет использован.
