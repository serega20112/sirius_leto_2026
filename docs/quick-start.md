# Быстрый старт

Ниже описан путь, который ближе всего к реальному использованию проекта на Windows.

## Что понадобится

- Python 3.10 или новее;
- рабочая камера или путь к видеофайлу;
- доступ в интернет хотя бы на первую загрузку моделей;
- если нужен GPU, то установленный `torch` со сборкой под вашу CUDA.

## Подготовка окружения

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements.ai.txt
```

Если планируется запуск распознавания лица на GPU через `facenet-pytorch`, лучше сразу поставить пакет отдельной командой:

```powershell
python -m pip install --no-deps "git+https://github.com/timesler/facenet-pytorch.git"
```

Это не случайный обходной путь. В некоторых сетях `PyPI` режется прокси, а GitHub остается доступен, поэтому прямой install из репозитория часто проходит стабильнее.

## Настройка `.env`

Минимальный рабочий вариант:

```env
CAMERA_SOURCE=0
LESSONS_BEGINNING=09:00
```

Если вместо камеры нужен файл:

```env
CAMERA_SOURCE=D:\video\lesson.mp4
```

Подробный список переменных вынесен в [configuration.md](configuration.md).

## Первый запуск

```powershell
python src/main.py
```

После старта приложение поднимается на `http://127.0.0.1:5000`.

Основные страницы:

- `/` — мониторинг и журнал;
- `/register` — регистрация ученика;
- `/groups` — список групп и карточки учеников.

## Минимальная проверка после запуска

1. Открыть `/register`.
2. Завести одного ученика с тремя фотографиями.
3. Вернуться на `/`.
4. Дождаться, пока имя начнет определяться на видео.
5. Проверить, что в `/groups` у ученика открывается история посещений.

## Если нужен именно GPU

Для проекта это означает три разных режима:

- `yolov8n.pt` и `yolov8n-pose.pt` должны идти через `torch` на `CUDA`;
- `facenet-pytorch` должен подняться без ошибки инициализации;
- запасной `DeepFace` в идеале не должен использоваться, потому что в типичной Windows-конфигурации он часто остается на CPU.

Понять, что реально произошло, можно по логам в консоли:

- `YOLO detector device: cuda`
- `YOLO pose device: cuda`
- `FaceNet backend: facenet_pytorch | device: ...`

Если вместо последней строки появляется `deepface | device: CPU`, стоит сразу открыть [troubleshooting.md](troubleshooting.md).
