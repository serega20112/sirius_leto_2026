from datetime import datetime, timedelta
from typing import Dict, Any

from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus


class TrackAttendanceUseCase:
    """Сценарий обработки видеокадра: детекция, распознавание, анализ вовлеченности."""

    def __init__(self, person_detector, face_recognizer, pose_estimator,
                 student_repo, attendance_repo):
        self.person_detector = person_detector
        self.face_recognizer = face_recognizer
        self.pose_estimator = pose_estimator
        self.student_repo = student_repo
        self.attendance_repo = attendance_repo

        self._log_cache = {}
        self._LOG_COOLDOWN = timedelta(seconds=60)
    def execute(self, frame) -> Dict[str, Any]:
        """
        Обрабатывает кадр.
        Возвращает:
          - processed_frame: кадр с отрисованными рамками
          - events: список событий (кто распознан)
        """
        bboxes = self.person_detector.detect_people(frame)

        active_students = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            person_img = frame[y1:y2, x1:x2]

            if person_img.size == 0 or person_img.shape[0] < 20 or person_img.shape[1] < 20:
                continue

            filename = self.face_recognizer.recognize(person_img)

            student_name = "Unknown"
            student_id = None
            engagement = "unknown"

            if filename:
                student_id = filename.split(".")[0]
                student = self.student_repo.find_by_id(student_id)

                if student:
                    student_name = student.name

                    engagement = self.pose_estimator.estimate_engagement(frame, bbox)

                    self._log_visit(student_id, engagement)

            active_students.append({
                "bbox": bbox,
                "name": student_name,
                "engagement": engagement
            })

        return {
            "students": active_students
        }

    def _log_visit(self, student_id: str, engagement: str):
        """Записывает лог, если прошло достаточно времени с последней записи."""
        now = datetime.now()
        last_time = self._log_cache.get(student_id)

        if last_time is None or (now - last_time) > self._LOG_COOLDOWN:
            lesson_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
            is_late = now > lesson_start

            status_enum = EngagementStatus(engagement) if engagement in ["high", "medium",
                                                                         "low"] else EngagementStatus.UNKNOWN

            log = AttendanceLog(
                id=None,
                student_id=student_id,
                timestamp=now,
                is_late=is_late,
                engagement_score=status_enum
            )

            self.attendance_repo.add_log(log)
            self._log_cache[student_id] = now