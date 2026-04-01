# HTTP API

Ниже перечислены роуты, которые реально используются проектом.

## Веб-страницы

| Метод | URL | Назначение |
| --- | --- | --- |
| `GET` | `/` | Главная страница мониторинга |
| `GET` | `/register` | Форма регистрации ученика |
| `GET` | `/groups` | Страница групп и карточек учеников |

## Данные и потоки

| Метод | URL | Назначение |
| --- | --- | --- |
| `GET` | `/video_feed` | MJPEG-поток с аннотациями поверх видео |
| `GET` | `/logs` | Журнал посещаемости для главной страницы |
| `GET` | `/students/<student_id>/attendance` | Детальная статистика посещаемости ученика |
| `GET` | `/src/assets/images/<filename>` | Миниатюра фотографии ученика |

## Изменяющие запросы

| Метод | URL | Назначение |
| --- | --- | --- |
| `POST` | `/register` | Регистрация нового ученика |
| `POST` | `/manual_status` | Заготовка для ручной отметки статуса |

## `POST /register`

Формат: `multipart/form-data`

Поля формы:

- `name`
- `group`
- `photos` — три файла с одним и тем же ключом

Пример ответа:

```json
{
  "id": "15ac14ef-e778-45cd-8699-e49a5aff8e70",
  "name": "Иван Петров",
  "group": "10A",
  "photo_paths": [
    "/src/assets/images/15ac14ef-e778-45cd-8699-e49a5aff8e70_1.jpg",
    "/src/assets/images/15ac14ef-e778-45cd-8699-e49a5aff8e70_2.jpg",
    "/src/assets/images/15ac14ef-e778-45cd-8699-e49a5aff8e70_3.jpg"
  ]
}
```

Если фото меньше или больше трех, use case вернет ошибку валидации.

## `GET /logs`

Пример ответа:

```json
[
  {
    "id": 12,
    "student_name": "Иван Петров",
    "timestamp": "2026-04-01T09:01:24.117321",
    "is_late": true,
    "status": "late",
    "arrived_at": "09:01",
    "engagement": "medium"
  }
]
```

Этот маршрут используется главной страницей. Фронт опрашивает его каждые две секунды.

## `GET /students/<student_id>/attendance`

Пример ответа:

```json
{
  "student": {
    "id": "15ac14ef-e778-45cd-8699-e49a5aff8e70",
    "name": "Иван Петров",
    "group": "10A"
  },
  "summary": {
    "lesson_days": 8,
    "attended_days": 7,
    "on_time_days": 6,
    "late_days": 1,
    "absent_days": 1,
    "attendance_rate": 88
  },
  "late_arrivals": [
    {
      "date": "2026-04-01",
      "arrived_at": "09:01"
    }
  ],
  "absences": [
    {
      "date": "2026-03-30"
    }
  ],
  "history": [
    {
      "date": "2026-04-01",
      "status": "late",
      "arrived_at": "09:01"
    }
  ]
}
```

## `POST /manual_status`

Формат: `application/json`

Минимальный payload:

```json
{
  "student_id": "15ac14ef-e778-45cd-8699-e49a5aff8e70",
  "status": "present"
}
```

Сейчас маршрут нужен скорее как техническая заготовка. Он проверяет, что поля пришли, и отвечает подтверждением. Новую запись в SQLite он пока не создает.

## Ошибки

Ошибки уровня валидации отдаются через общие error handlers из `src/backend/delivery/error_handlers.py`. Поэтому frontend обычно получает нормальный JSON, а не HTML-трейсбек.
