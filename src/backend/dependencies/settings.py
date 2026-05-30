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

    # YOLO models removed for edge/MediaPipe-only deployment
    # Paths for YOLOv8 person detection model (OpenVINO IR format)
    YOLO_MODEL_PATH = ""
    # Path for YOLOv8 pose model (not used in this stack, kept for compatibility)
    YOLO_POSE_MODEL_PATH = ""
    # Path for YOLOv8 person detection model (OpenVINO IR format) – used in stack 8
    YOLOV8_MODEL_PATH = str(ASSETS_DIR / "models" / "yolov8n_person.xml")
    # Path for SCRFD face detection model (OpenVINO IR format) – used in stack 8
    SCRFD_MODEL_PATH = str(ASSETS_DIR / "models" / "scrfd.xml")
    # Path for RTMPose full-body pose estimation model (OpenVINO IR format) – used in stack 8
    RTMPose_MODEL_PATH = str(ASSETS_DIR / "models" / "rtmpose.xml")
    MP_FACE_LANDMARKER_MODEL_PATH = str(ASSETS_DIR / "models" / "face_landmarker.task")
    MP_POSE_LANDMARKER_MODEL_PATH = str(
        ASSETS_DIR / "models" / "pose_landmarker_full.task"
    )
    DB_PATH = ASSETS_DIR / "database" / "attendance.db"

    CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", 0)

    # Edge device tuning (Raspberry Pi / low-power)
    CAMERA_WIDTH = _get_int("CAMERA_WIDTH", 640)
    CAMERA_HEIGHT = _get_int("CAMERA_HEIGHT", 480)
    # Target streaming FPS for edge devices (e.g., Raspberry Pi). Adjusted to achieve ~20 FPS.
    STREAM_FPS = _get_float("STREAM_FPS", 20.0)
    # Inference rate (how many inference cycles per second). Set to match target FPS.
    INFERENCE_RATE = _get_float("INFERENCE_RATE", 20.0)
    EDGE_MODE = str(_get_env_value("EDGE_MODE", default="false")).strip().lower() in (
        "1",
        "true",
        "yes",
    )

    # Prefer lightweight person detector on edge devices by default
    USE_LIGHT_PERSON_DETECTOR = str(
        _get_env_value(
            "USE_LIGHT_PERSON_DETECTOR", default="true" if EDGE_MODE else "false"
        )
    ).strip().lower() in ("1", "true", "yes")

    # Paths for lightweight detection models (optional)
    MOBILENET_SSD_PROTOTXT = str(ASSETS_DIR / "models" / "MobileNetSSD_deploy.prototxt")
    MOBILENET_SSD_CAFFEMODEL = str(
        ASSETS_DIR / "models" / "MobileNetSSD_deploy.caffemodel"
    )
    MOBILENET_SSD_ONNX = str(ASSETS_DIR / "models" / "mobilenet_ssd.onnx")
    # YuNet face detector (OpenCV DNN ONNX). Provides fast and accurate face detection.
    # The model can be downloaded from OpenCV model zoo: https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
    YUNET_MODEL_PATH = str(ASSETS_DIR / "models" / "face_detection_yunet_2022mar.onnx")

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

    # Lower defaults for MediaPipe to increase detection sensitivity on edge devices
    MP_MIN_DETECTION_CONFIDENCE = _get_float(
        "MP_MIN_DETECTION_CONFIDENCE",
        0.35,
    )
    MP_MIN_TRACKING_CONFIDENCE = _get_float(
        "MP_MIN_TRACKING_CONFIDENCE",
        0.35,
    )
    MP_SMOOTHING_WINDOW = _get_int("MP_SMOOTHING_WINDOW", 5)
    MP_HIGH_THRESHOLD = _get_float("MP_HIGH_THRESHOLD", 0.72)
    MP_MEDIUM_THRESHOLD = _get_float("MP_MEDIUM_THRESHOLD", 0.45)

    # Person filtering heuristics to reduce false positives (hands, small objects)
    MIN_PERSON_HEIGHT_RATIO = _get_float("MIN_PERSON_HEIGHT_RATIO", 0.12)
    MIN_PERSON_AREA_RATIO = _get_float("MIN_PERSON_AREA_RATIO", 0.003)
    MIN_PERSON_ASPECT_RATIO = _get_float("MIN_PERSON_ASPECT_RATIO", 0.25)
    MAX_PERSON_ASPECT_RATIO = _get_float("MAX_PERSON_ASPECT_RATIO", 2.5)

    # Engagement evaluation throttling (seconds) to reduce CPU usage
    ENGAGEMENT_EVAL_INTERVAL = _get_float("ENGAGEMENT_EVAL_INTERVAL", 0.35)

    # Processing sizes for face / person crops to speed up inference
    FACE_PROCESS_SIZE = _get_int("FACE_PROCESS_SIZE", 160)
    PERSON_PROCESS_SIZE = _get_int("PERSON_PROCESS_SIZE", 320)

    # Motion-based confirmation for detections without face
    MIN_TRACK_CONFIRMED_FRAMES = _get_int("MIN_TRACK_CONFIRMED_FRAMES", 2)
    MOTION_DIFF_THRESHOLD = _get_int("MOTION_DIFF_THRESHOLD", 20)
    MOTION_MIN_ACTIVITY_RATIO = _get_float("MOTION_MIN_ACTIVITY_RATIO", 0.02)

    # Engagement score weights (configurable)
    ENGAGEMENT_FACE_WEIGHT = _get_float("ENGAGEMENT_FACE_WEIGHT", 0.7)
    ENGAGEMENT_BODY_WEIGHT = _get_float("ENGAGEMENT_BODY_WEIGHT", 0.3)

    # Debugging helpers
    DEBUG_DETECTION = str(
        _get_env_value("DEBUG_DETECTION", default="false")
    ).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    DEBUG_SAVE_FAILING_FRAMES = str(
        _get_env_value("DEBUG_SAVE_FAILING_FRAMES", default="false")
    ).strip().lower() in ("1", "true", "yes")
    DEBUG_SAVE_DIR = ASSETS_DIR / "debug_failures"

    # Bounding box smoothing alpha (EWMA)
    BBOX_SMOOTH_ALPHA = _get_float("BBOX_SMOOTH_ALPHA", 0.45)

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

    # Detection frequency control – skip this many frames between full detections.
    # Setting to 0 means run detection on every frame.
    DETECTION_SKIP_FRAMES = _get_int("DETECTION_SKIP_FRAMES", 0)


settings = Settings()
