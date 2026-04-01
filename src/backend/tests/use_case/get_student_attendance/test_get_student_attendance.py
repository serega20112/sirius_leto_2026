from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.backend.application.exceptions import ValidationError
from src.backend.domain.attendance.entity import AttendanceLog, EngagementStatus
from src.backend.use_case.get_student_attendance import GetStudentAttendanceUseCase


def _build_log(student_id: str, timestamp: datetime, *, is_late: bool) -> AttendanceLog:
    """
    Verifies scenario build log.
    
    Args:
        student_id: Input value for `student_id`.
        timestamp: Input value for `timestamp`.
        is_late: Input value for `is_late`.
    
    Returns:
        The function result.
    """
    return AttendanceLog(
        id=None,
        student_id=student_id,
        timestamp=timestamp,
        is_late=is_late,
        engagement_score=EngagementStatus.UNKNOWN,
    )


def test_execute_builds_student_attendance_stats():
    """
    Verifies scenario execute builds student attendance stats.
    
    Args:
        None.
    
    Returns:
        Does not return a value.
    """
    student = SimpleNamespace(id="student-1", name="Alice", group_name="10A")
    classmates = [
        student,
        SimpleNamespace(id="student-2", name="Bob", group_name="10A"),
        SimpleNamespace(id="student-3", name="Carol", group_name="11B"),
    ]

    attendance_repo = MagicMock()
    attendance_repo.get_all_logs.return_value = [
        _build_log("student-1", datetime(2026, 4, 1, 9, 1, 0), is_late=False),
        _build_log("student-2", datetime(2026, 4, 2, 9, 0, 0), is_late=False),
        _build_log("student-1", datetime(2026, 4, 3, 9, 17, 0), is_late=True),
        _build_log("student-3", datetime(2026, 4, 4, 9, 0, 0), is_late=False),
    ]

    student_repo = MagicMock()
    student_repo.find_by_id.return_value = student
    student_repo.get_all.return_value = classmates

    use_case = GetStudentAttendanceUseCase(attendance_repo, student_repo)
    result = use_case.execute("student-1")

    assert result["student"] == {"id": "student-1", "name": "Alice", "group": "10A"}
    assert result["summary"] == {
        "lesson_days": 3,
        "attended_days": 2,
        "on_time_days": 1,
        "late_days": 1,
        "absent_days": 1,
        "attendance_rate": 67,
    }
    assert result["late_arrivals"] == [{"date": "2026-04-03", "arrived_at": "09:17"}]
    assert result["absences"] == [{"date": "2026-04-02"}]
    assert result["history"] == [
        {"date": "2026-04-03", "status": "late", "arrived_at": "09:17"},
        {"date": "2026-04-02", "status": "absent", "arrived_at": None},
        {"date": "2026-04-01", "status": "present", "arrived_at": "09:01"},
    ]


def test_execute_raises_for_unknown_student():
    """
    Verifies scenario execute raises for unknown student.
    
    Args:
        None.
    
    Returns:
        Does not return a value.
    """
    attendance_repo = MagicMock()
    student_repo = MagicMock()
    student_repo.find_by_id.return_value = None

    use_case = GetStudentAttendanceUseCase(attendance_repo, student_repo)

    with pytest.raises(ValidationError, match="Ученик не найден"):
        use_case.execute("missing-student")
