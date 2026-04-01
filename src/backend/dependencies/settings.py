from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    BACKEND_ROOT = Path(__file__).resolve().parent.parent
    ASSETS_DIR = BACKEND_ROOT / "assets"

    IMAGES_DIR = ASSETS_DIR / "images"

    YOLO_MODEL_PATH = str(ASSETS_DIR / "models" / "yolov8n.pt")
    YOLO_POSE_MODEL_PATH = str(ASSETS_DIR / "models" / "yolov8n-pose.pt")
    DB_PATH = ASSETS_DIR / "database" / "attendance.db"

    CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", 0)

    FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "Facenet512")
    FACE_DETECTOR_BACKEND = os.getenv("FACE_DETECTOR_BACKEND", "retinaface")
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


settings = Settings()
