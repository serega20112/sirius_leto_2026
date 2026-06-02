from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True)
class FaceRecognitionConfig:
    model_name: str = "Facenet512"
    detector_backend: str = "retinaface"
    normalization: str = "Facenet2018"
    runtime_backend: str = "auto"
    device: str = "auto"
    embedding_model_name: str = "vggface2"
    embedding_image_size: int = 160
    embedding_margin: int = 0
    min_face_confidence: float = 0.80
    distance_threshold: float = 0.50
    min_margin: float = 0.02
    min_face_size: int = 80
    min_stable_votes: int = 2
    vote_window: int = 5


@dataclass(frozen=True)
class EngagementConfig:
    head_crop_ratio: float = 0.55
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smoothing_window: int = 5
    high_threshold: float = 0.72
    medium_threshold: float = 0.45
    max_yaw_degrees: float = 40.0
    max_pitch_degrees: float = 25.0
    min_body_visibility: float = 0.4


@dataclass
class AttendanceTrackingConfig:
    """Configuration for attendance tracking.

    The original configuration only contained a lesson start time. To support
    explicit lesson end times we add ``lesson_end_time``. The dataclass is no
    longer frozen so the values can be updated at runtime via the new API
    endpoint.
    """

    presence_confirmation_seconds: float = 3.0
    log_cooldown_seconds: float = 60.0
    late_after_seconds: float = 60.0
    stale_track_ttl_seconds: float = 10.0
    face_crop_width_ratio: float = 0.7
    face_crop_height_ratio: float = 0.5
    lesson_start_time: time = time(9, 0)
    # New attribute: end of lesson. Default to one hour after start.
    lesson_end_time: time = time(10, 0)
