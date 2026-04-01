from src.backend.infrastructure.database import SessionLocal
from src.backend.application.services import (
    AttendanceApplicationService,
    MediaApplicationService,
    StudentApplicationService,
)
from src.backend.use_case.get_report import GetReportUseCase
from src.backend.use_case.get_student_attendance import GetStudentAttendanceUseCase
from src.backend.use_case.get_groups import GetGroupsUseCase
from src.backend.use_case.register_student import RegisterStudentUseCase
from src.backend.use_case.track_attendance import TrackAttendanceUseCase
from src.backend.infrastructure.ai.config import (
    AttendanceTrackingConfig,
    EngagementConfig,
    FaceRecognitionConfig,
)
from src.backend.infrastructure.storage.local_files import LocalFileStorage
from src.backend.infrastructure.media import StudentPhotoProvider
from src.backend.infrastructure.ai.face.recognizer import FaceRecognizer
from src.backend.infrastructure.persistence.sqlite import (
    SqliteAttendanceRepository,
    SqliteStudentRepository,
)
from src.backend.infrastructure.video.annotated_streamer import AnnotatedVideoStreamer
from src.backend.dependencies import settings

from src.backend.infrastructure.ai.person.detector import PersonDetector
from src.backend.infrastructure.ai.person.pose import (
    PoseEstimator as PersonPoseEstimator,
)


class Container:
    def __init__(self):
        self.session = SessionLocal()
        self.student_repository = SqliteStudentRepository(self.session)
        self.attendance_repository = SqliteAttendanceRepository(self.session)
        self.file_storage = LocalFileStorage(str(settings.IMAGES_DIR))
        self.student_photo_provider = StudentPhotoProvider(settings.IMAGES_DIR)
        self.face_recognition_config = FaceRecognitionConfig(
            model_name=settings.FACE_MODEL_NAME,
            runtime_backend=settings.FACE_RUNTIME_BACKEND,
            device=settings.AI_DEVICE,
            detector_backend=settings.FACE_DETECTOR_BACKEND,
            embedding_model_name=settings.FACE_EMBEDDING_MODEL_NAME,
            embedding_image_size=settings.FACE_EMBEDDING_IMAGE_SIZE,
            embedding_margin=settings.FACE_EMBEDDING_MARGIN,
            min_face_confidence=settings.FACE_MIN_DETECTION_CONFIDENCE,
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
            lesson_start_time=settings.LESSON_START_TIME,
        )
        self.face_recognizer = FaceRecognizer(
            str(settings.IMAGES_DIR),
            config=self.face_recognition_config,
        )
        self.register_student_use_case = RegisterStudentUseCase(
            self.student_repository,
            self.file_storage,
            self.face_recognizer,
        )
        self.get_groups_use_case = GetGroupsUseCase(self.student_repository)
        self.get_report_use_case = GetReportUseCase(
            self.attendance_repository,
            self.student_repository,
        )
        self.get_student_attendance_use_case = GetStudentAttendanceUseCase(
            self.attendance_repository,
            self.student_repository,
        )

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

        self.track_attendance_use_case = TrackAttendanceUseCase(
            self.person_detector,
            self.face_recognizer,
            self.pose_estimator,
            self.student_repository,
            self.attendance_repository,
            config=self.attendance_tracking_config,
        )
        self.video_streamer = AnnotatedVideoStreamer(self.track_attendance_use_case)
        self.student_service = StudentApplicationService(
            self.register_student_use_case,
            self.get_groups_use_case,
        )
        self.attendance_service = AttendanceApplicationService(
            self.video_streamer,
            self.get_report_use_case,
            self.get_student_attendance_use_case,
        )
        self.media_service = MediaApplicationService(self.student_photo_provider)
