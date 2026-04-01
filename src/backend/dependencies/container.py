from src.backend.infrastructure.database import SessionLocal
from src.backend.repository.attendance.attendance_repo import SqliteAttendanceRepository
from src.backend.repository.student.student_repo import SqliteStudentRepository
from src.backend.services.attendance_service import AttendanceService
from src.backend.services.monitor_service import MonitorService
from src.backend.services.register_service import RegisterService
from src.backend.services.student_service import StudentService
from src.backend.use_case.get_report import GetReportUseCase
from src.backend.use_case.register_student import RegisterStudentUseCase
from src.backend.use_case.track_attendance import TrackAttendanceUseCase
from src.backend.infrastructure.ai.config import (
    AttendanceTrackingConfig,
    EngagementConfig,
    FaceRecognitionConfig,
)
from src.backend.infrastructure.storage.local_files import LocalFileStorage
from src.backend.infrastructure.ai.face.recognizer import FaceRecognizer
from src.backend.dependencies import settings

from src.backend.infrastructure.ai.person.detector import PersonDetector
from src.backend.infrastructure.ai.person.pose import (
    PoseEstimator as PersonPoseEstimator,
)


class Container:
    def __init__(self):
        self.session = SessionLocal()
        self.student_repo = SqliteStudentRepository(self.session)
        self.attendance_repo = SqliteAttendanceRepository(self.session)
        self.student_service = StudentService(self.student_repo)
        self.attendance_service = AttendanceService(self.attendance_repo)
        self.file_storage = LocalFileStorage(str(settings.IMAGES_DIR))
        self.face_recognition_config = FaceRecognitionConfig(
            model_name=settings.FACE_MODEL_NAME,
            detector_backend=settings.FACE_DETECTOR_BACKEND,
            distance_threshold=settings.FACE_DISTANCE_THRESHOLD,
            min_margin=settings.FACE_DISTANCE_MARGIN,
            min_stable_votes=settings.FACE_MIN_STABLE_VOTES,
            vote_window=settings.FACE_VOTE_WINDOW,
        )
        self.engagement_config = EngagementConfig(
            min_detection_confidence=settings.MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.MP_MIN_TRACKING_CONFIDENCE,
            smoothing_window=settings.MP_SMOOTHING_WINDOW,
            high_threshold=settings.MP_HIGH_THRESHOLD,
            medium_threshold=settings.MP_MEDIUM_THRESHOLD,
        )
        self.attendance_tracking_config = AttendanceTrackingConfig(
            presence_confirmation_seconds=settings.PRESENCE_CONFIRMATION_SECONDS,
            log_cooldown_seconds=settings.ATTENDANCE_LOG_COOLDOWN_SECONDS,
            late_after_seconds=settings.ATTENDANCE_LATE_AFTER_SECONDS,
            stale_track_ttl_seconds=settings.STALE_TRACK_TTL_SECONDS,
        )
        self.face_recognizer = FaceRecognizer(
            str(settings.IMAGES_DIR),
            config=self.face_recognition_config,
        )
        self.register_use_case = RegisterStudentUseCase(
            self.student_repo, self.file_storage, self.face_recognizer
        )
        self.register_service = RegisterService(self.register_use_case)
        self.get_report_use_case = GetReportUseCase(
            self.attendance_repo, self.student_repo
        )

        # Инициализация детектора и pose estimator
        try:
            self.person_detector = PersonDetector(settings.YOLO_MODEL_PATH)
            print("[Container] person_detector initialized")
        except Exception as e:
            print(f"[Container] person_detector init error: {e}")
            self.person_detector = None

        try:
            self.pose_estimator = PersonPoseEstimator(config=self.engagement_config)
            print("[Container] pose_estimator initialized")
        except Exception as e:
            print(f"[Container] pose_estimator init error: {e}")
            self.pose_estimator = None

        # TrackAttendanceUseCase получает работающие реализации
        self.track_attendance_use_case = TrackAttendanceUseCase(
            self.person_detector,
            self.face_recognizer,
            self.pose_estimator,
            self.student_repo,
            self.attendance_repo,
            config=self.attendance_tracking_config,
        )
        self.monitor_service = MonitorService(
            self.track_attendance_use_case,
            self.get_report_use_case,
            self.student_service,
        )


container = Container()


def get_monitor_service():
    monitor_service = container.monitor_service
    return monitor_service


def get_student_service():
    student_service = container.student_service
    return student_service


def get_register_service():
    register_service = container.register_service
    return register_service
