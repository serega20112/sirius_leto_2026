from src.backend.infrastructure.database import SessionLocal
from src.backend.infrastructure.storage.local_files import LocalFileStorage
from src.backend.infrastructure.ai.person.detector import PersonDetector
from src.backend.infrastructure.ai.person.pose import PoseEstimator
from src.backend.infrastructure.ai.face.recognizer import FaceRecognizer
from src.backend.repository import SqliteStudentRepository
from src.backend.repository import SqliteAttendanceRepository
from src.backend.use_case.register_student import RegisterStudentUseCase
from src.backend.use_case.track_attendance import TrackAttendanceUseCase
from src.backend.use_case.get_report import GetReportUseCase
from src.backend.dependencies import Settings


class Container:
    """DI Контейнер для сборки зависимостей."""

    def __init__(self):
        self.settings = Settings()
        self.db_session = SessionLocal()

        self.storage = LocalFileStorage(base_path=self.settings.IMAGES_DIR)

        self.person_detector = PersonDetector(model_path=self.settings.YOLO_MODEL_PATH)
        self.face_recognizer = FaceRecognizer(db_path=self.settings.IMAGES_DIR)
        self.pose_estimator = PoseEstimator()

        self.student_repo = SqliteStudentRepository(self.db_session)
        self.attendance_repo = SqliteAttendanceRepository(self.db_session)

        self.register_use_case = RegisterStudentUseCase(
            student_repo=self.student_repo,
            file_storage=self.storage,
            face_recognizer=self.face_recognizer
        )

        self.track_use_case = TrackAttendanceUseCase(
            person_detector=self.person_detector,
            face_recognizer=self.face_recognizer,
            pose_estimator=self.pose_estimator,
            student_repo=self.student_repo,
            attendance_repo=self.attendance_repo
        )

        self.report_use_case = GetReportUseCase(
            attendance_repo=self.attendance_repo,
            student_repo=self.student_repo
        )


container = Container()