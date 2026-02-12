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
from src.backend.infrastructure.storage.local_files import LocalFileStorage
from src.backend.infrastructure.ai.face.recognizer import FaceRecognizer
from src.backend.dependencies import settings

from src.backend.infrastructure.ai.person.detector import PersonDetector
from src.backend.infrastructure.ai.person.pose import PoseEstimator as PersonPoseEstimator


class Container:
    def __init__(self):
        self.session = SessionLocal()
        self.student_repo = SqliteStudentRepository(self.session)
        self.attendance_repo = SqliteAttendanceRepository(self.session)
        self.student_service = StudentService(self.student_repo)
        self.attendance_service = AttendanceService(self.attendance_repo)
        self.file_storage = LocalFileStorage(str(settings.IMAGES_DIR))
        self.face_recognizer = FaceRecognizer(str(settings.IMAGES_DIR))
        self.register_use_case = RegisterStudentUseCase(self.student_repo, self.file_storage, self.face_recognizer)
        self.register_service = RegisterService(self.register_use_case)
        self.get_report_use_case = GetReportUseCase(self.attendance_repo, self.student_repo)

        # Инициализация детектора и pose estimator
        try:
            self.person_detector = PersonDetector(settings.YOLO_MODEL_PATH)
            print('[Container] person_detector initialized')
        except Exception as e:
            print(f'[Container] person_detector init error: {e}')
            self.person_detector = None

        try:
            self.pose_estimator = PersonPoseEstimator()
            print('[Container] pose_estimator initialized')
        except Exception as e:
            print(f'[Container] pose_estimator init error: {e}')
            self.pose_estimator = None

        # TrackAttendanceUseCase получает работающие реализации
        self.track_attendance_use_case = TrackAttendanceUseCase(self.person_detector, self.face_recognizer, self.pose_estimator, self.student_repo, self.attendance_repo)
        self.monitor_service = MonitorService(self.track_attendance_use_case, self.get_report_use_case, self.student_service)


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
