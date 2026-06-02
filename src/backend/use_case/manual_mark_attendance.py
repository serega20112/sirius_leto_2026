from datetime import datetime

from src.backend.application.exceptions import ValidationError
from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus


class ManualMarkAttendanceUseCase:
    """Use-case to create manual attendance logs (teacher/admin overrides)."""

    def __init__(self, attendance_repository, student_repository, config=None):
        self.attendance_repository = attendance_repository
        self.student_repository = student_repository
        self.config = config

    def execute(self, student_id: str, status: str) -> dict:
        if not student_id:
            raise ValidationError("Нужен student_id")

        student = self.student_repository.find_by_id(student_id)
        if student is None:
            raise ValidationError("Ученик не найден")

        if status not in ("present", "late"):
            raise ValidationError("Неподдерживаемый статус. Используйте 'present' или 'late'.")

        is_late = status == "late"
        now = datetime.now()

        log = AttendanceLog(
            id=None,
            student_id=student_id,
            timestamp=now,
            is_late=is_late,
            engagement_score=EngagementStatus.UNKNOWN,
        )

        created = self.attendance_repository.add_log(log)

        return {
            "status": "created",
            "student_id": student_id,
            "is_late": created.is_late,
            "timestamp": created.timestamp.isoformat(),
            "log_id": created.id,
        }
