# Архитектура

Проект собран вокруг довольно прямой схемы: delivery принимает HTTP-запрос, application-service переводит его в язык сценариев, use case отрабатывает бизнес-правило, а infrastructure дает доступ к базе, файлам, видео и AI.

## Карта каталогов

```text
src/
  backend/
    application/
      services/
    delivery/
      api/v1/
    dependencies/
    domain/
      attendance/
      student/
    infrastructure/
      ai/
      media/
      persistence/sqlite/
      storage/
      video/
    use_case/
  frontend/
    static/
    templates/
```

## Что относится к какому слою

### Delivery

Файлы:

- `src/backend/delivery/api/v1/index_route.py`
- `src/backend/delivery/api/v1/auth_route.py`
- `src/backend/delivery/api/v1/monitor_route.py`
- `src/backend/delivery/api/v1/media_route.py`

Роуты здесь специально тонкие. Они получают данные из `request`, вызывают application-service и возвращают `jsonify`, `render_template` или `Response`.

### Application

Файлы:

- `src/backend/application/services/student_service.py`
- `src/backend/application/services/attendance_service.py`
- `src/backend/application/services/media_service.py`

Этот слой нужен не ради красоты. Через него delivery перестает знать, как именно устроены use case и в каком формате backend хранит данные. Здесь же происходит простая сериализация доменных объектов в payload для фронта.

### Use case

Файлы:

- `src/backend/use_case/register_student.py`
- `src/backend/use_case/track_attendance.py`
- `src/backend/use_case/get_groups.py`
- `src/backend/use_case/get_report.py`
- `src/backend/use_case/get_student_attendance.py`

Это слой сценариев. Здесь уже есть правила, которые относятся к предметной области:

- при регистрации нужно ровно три фото;
- ученик считается пришедшим только после нескольких секунд в кадре;
- опоздание считается от времени начала урока;
- пропуск в карточке ученика вычисляется по истории группы.

### Domain

Файлы:

- `src/backend/domain/student/entity.py`
- `src/backend/domain/student/repository.py`
- `src/backend/domain/attendance/entity.py`
- `src/backend/domain/attendance/repository.py`

Тут лежат сущности и контракты репозиториев. Это самый устойчивый слой: он не знает ни про Flask, ни про SQLAlchemy, ни про OpenCV.

### Infrastructure

Файлы:

- `src/backend/infrastructure/persistence/sqlite/*`
- `src/backend/infrastructure/storage/local_files.py`
- `src/backend/infrastructure/media/student_photo_provider.py`
- `src/backend/infrastructure/video/annotated_streamer.py`
- `src/backend/infrastructure/ai/**/*`

Здесь находится все, что связано с конкретной технологией: SQLite, файловая система, работа с кадрами, модели `YOLO`, `FaceNet`, `MediaPipe` и так далее.

## Как выглядит типовой запрос

### Регистрация ученика

1. `POST /register` принимает форму и файлы.
2. `StudentApplicationService.register_student` передает данные в use case.
3. `RegisterStudentUseCase.execute` валидирует вход, создает `Student`, сохраняет файлы и обновляет face gallery.
4. SQLite-репозиторий сохраняет сущность.
5. Route возвращает JSON с публичными ссылками на фотографии.

### Живой мониторинг

1. `/video_feed` отдает generator из `AnnotatedVideoStreamer`.
2. Поток читает кадры и на каждом кадре вызывает `TrackAttendanceUseCase`.
3. `TrackAttendanceUseCase` использует:
   - `PersonDetector` для людей;
   - `FaceRecognizer` для личности;
   - `PoseEstimator` для вовлеченности.
4. После подтверждения присутствия в `AttendanceRepository` пишется `AttendanceLog`.
5. Готовый кадр уходит обратно во frontend как MJPEG stream.

## Где происходит сборка зависимостей

`src/backend/dependencies/container.py` — это точка, где все связывается друг с другом. Он создает:

- SQLite-репозитории;
- file storage;
- AI-конфиги;
- AI-адаптеры;
- use case;
- application-service;
- video streamer.

`src/backend/create_app.py` использует контейнер при сборке Flask-приложения и передает нужные сервисы в blueprints.

## Что здесь пока не доведено до идеала

- use case лежат в одном каталоге `src/backend/use_case`, а не разнесены по bounded context;
- `manual_status` пока не проводит изменения в доменную модель;
- в `src/backend/infrastructure/database.py` SQLAlchemy-модели и bootstrap базы живут в одном файле;
- посещаемость считается от факта логов, а не от отдельной сущности расписания.

Это не ломает работу, но эти точки стоит иметь в виду, если проект будет расти дальше.
