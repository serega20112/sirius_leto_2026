import os
from datetime import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_env_value(*names: str, default=None):
    """
    Runs the internal step get env value.
    
    Args:
        default: Input value for `default`.
        *names: Input value for `*names`.
    
    Returns:
        The function result.
    """
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _get_float(name: str, default: float) -> float:
    """
    Runs the internal step get float.
    
    Args:
        name: Input value for `name`.
        default: Input value for `default`.
    
    Returns:
        The function result.
    """
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    """
    Runs the internal step get int.
    
    Args:
        name: Input value for `name`.
        default: Input value for `default`.
    
    Returns:
        The function result.
    """
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_time(names: tuple[str, ...], default: str) -> time:
    """
    Runs the internal step get time.
    
    Args:
        names: Input value for `names`.
        default: Input value for `default`.
    
    Returns:
        The function result.
    """
    raw_value = _get_env_value(*names, default=default)
    try:
        hours_text, minutes_text = str(raw_value).strip().split(":", maxsplit=1)
        return time(hour=int(hours_text), minute=int(minutes_text))
    except (TypeError, ValueError):
        default_hours, default_minutes = default.split(":", maxsplit=1)
        return time(hour=int(default_hours), minute=int(default_minutes))


class Settings:
    BACKEND_ROOT = Path(__file__).resolve().parent.parent
    ASSETS_DIR = BACKEND_ROOT / "assets"

    IMAGES_DIR = ASSETS_DIR / "images"

    YOLO_MODEL_PATH = str(ASSETS_DIR / "models" / "yolov8n.pt")
    YOLO_POSE_MODEL_PATH = str(ASSETS_DIR / "models" / "yolov8n-pose.pt")
    MP_FACE_LANDMARKER_MODEL_PATH = str(
        ASSETS_DIR / "models" / "face_landmarker.task"
    )
    MP_POSE_LANDMARKER_MODEL_PATH = str(
        ASSETS_DIR / "models" / "pose_landmarker_full.task"
    )
    DB_PATH = ASSETS_DIR / "database" / "attendance.db"

    CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", 0)

    AI_DEVICE = os.getenv("AI_DEVICE", "auto")
    FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "Facenet512")
    FACE_RUNTIME_BACKEND = os.getenv("FACE_RUNTIME_BACKEND", "auto")
    FACE_DETECTOR_BACKEND = os.getenv("FACE_DETECTOR_BACKEND", "retinaface")
    FACE_EMBEDDING_MODEL_NAME = os.getenv("FACE_EMBEDDING_MODEL_NAME", "vggface2")
    FACE_EMBEDDING_IMAGE_SIZE = _get_int("FACE_EMBEDDING_IMAGE_SIZE", 160)
    FACE_EMBEDDING_MARGIN = _get_int("FACE_EMBEDDING_MARGIN", 0)
    FACE_MIN_DETECTION_CONFIDENCE = _get_float(
        "FACE_MIN_DETECTION_CONFIDENCE",
        0.80,
    )
    FACE_DISTANCE_THRESHOLD = _get_float("FACE_DISTANCE_THRESHOLD", 0.50)
    FACE_DISTANCE_MARGIN = _get_float("FACE_DISTANCE_MARGIN", 0.02)
    FACE_MIN_STABLE_VOTES = _get_int("FACE_MIN_STABLE_VOTES", 2)
    FACE_VOTE_WINDOW = _get_int("FACE_VOTE_WINDOW", 5)

    MP_MIN_DETECTION_CONFIDENCE = _get_float(
        "MP_MIN_DETECTION_CONFIDENCE",
        0.5,
    )
    MP_MIN_TRACKING_CONFIDENCE = _get_float(
        "MP_MIN_TRACKING_CONFIDENCE",
        0.5,
    )
    MP_SMOOTHING_WINDOW = _get_int("MP_SMOOTHING_WINDOW", 5)
    MP_HIGH_THRESHOLD = _get_float("MP_HIGH_THRESHOLD", 0.72)
    MP_MEDIUM_THRESHOLD = _get_float("MP_MEDIUM_THRESHOLD", 0.45)

    PRESENCE_CONFIRMATION_SECONDS = _get_float(
        "PRESENCE_CONFIRMATION_SECONDS",
        3.0,
    )
    ATTENDANCE_LOG_COOLDOWN_SECONDS = _get_float(
        "ATTENDANCE_LOG_COOLDOWN_SECONDS",
        60.0,
    )
    ATTENDANCE_LATE_AFTER_SECONDS = _get_float(
        "ATTENDANCE_LATE_AFTER_SECONDS",
        60.0,
    )
    STALE_TRACK_TTL_SECONDS = _get_float("STALE_TRACK_TTL_SECONDS", 10.0)
    LESSON_START_TIME = _get_time(
        ("LESSONS_BEGINNING", "LESSON_START_TIME", "lessons_begining"),
        "09:00",
    )


settings = Settings()
